#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <filesystem>    // C++17
#include "../Chead/json.hpp"      // nlohmann/json (請先安裝或放置對應單檔於同目錄)

namespace fs = std::filesystem;
using json = nlohmann::json;

// ----------------------------------------------------------
// 1) 定義資料結構：儲存每一筆 (原始) 測站紀錄
//    為簡化，我們只關注 Python 腳本中最常用的欄位
// ----------------------------------------------------------
struct StationRecord {
    // 必要欄位 (在 Python 會以 -99 代表缺失)
    double air_temperature;
    double relative_humidity;
    double precipitation;
    double wind_speed;

    // 觀測時間
    std::string date_time; 

    // 其他欄位(例如 station_id, weather, wind_direction...) 若需要可再擴充
    // 這裡示範只保留 Python 腳本中最常被使用到的
};

// 這裡是從 JSON 讀取數值，若不存在或型別不符，則回傳預設值
double safeGetNumber(const json& j, const std::string& key, double default_val) {
    if (!j.contains(key) || j[key].is_null() || !j[key].is_number()) {
        return default_val;
    }
    return j[key].get<double>();
}

// ----------------------------------------------------------
// 2) 讀取測站 JSON -> 轉為 StationRecord 陣列
//    格式假設與先前 Python 拆分後的 station_id.json 相符
// ----------------------------------------------------------
bool load_station_data(const std::string& filepath, std::vector<StationRecord>& records) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        std::cerr << "無法開啟檔案: " << filepath << std::endl;
        return false;
    }

    json j;
    try {
        ifs >> j;
    } catch (...) {
        std::cerr << "解析 JSON 失敗: " << filepath << std::endl;
        return false;
    }

    // 假設 station_id.json 的結構是每筆資料都包含
    // {
    //   "id": ...,
    //   "station_id": "...",
    //   "air_temperature": ...,
    //   "relative_humidity": ...,
    //   "precipitation": ...,
    //   "wind_speed": ...,
    //   "date_time": "YYYY-MM-DD HH:MM:SS"
    // }
    // 實務上請依實際欄位名稱調整

    if (!j.is_array()) {
        std::cerr << "JSON 結構非陣列: " << filepath << std::endl;
        return false;
    }

    for (auto& item : j) {
        StationRecord rec;

        // 如果有缺欄位，則預設 -99
        rec.air_temperature     = safeGetNumber(item, "air_temperature", -99.0);
        rec.relative_humidity   = safeGetNumber(item, "relative_humidity", -99.0);
        rec.precipitation       = safeGetNumber(item, "precipitation", -99.0);
        rec.wind_speed          = safeGetNumber(item, "wind_speed", -99.0);
        rec.date_time         = item.value("date_time", "");

        records.push_back(rec);
    }
    return true;
}

// ----------------------------------------------------------
// 3) 資料清洗：將 -99 替換為 NaN，然後執行線性插值 & 前後向補值
//    這裡僅示範針對 air_temperature、relative_humidity、precipitation、wind_speed
// ----------------------------------------------------------
static const double MISSING_VALUE = -99.0;

void replace_missing_with_nan(std::vector<StationRecord>& records) {
    auto to_nan_if_missing = [](double x) {
        return (std::fabs(x - MISSING_VALUE) < 1e-9) ? std::numeric_limits<double>::quiet_NaN() : x;
    };
    for (auto& r : records) {
        r.air_temperature   = to_nan_if_missing(r.air_temperature);
        r.relative_humidity = to_nan_if_missing(r.relative_humidity);
        r.precipitation     = to_nan_if_missing(r.precipitation);
        r.wind_speed        = to_nan_if_missing(r.wind_speed);
    }
}

// 線性插值的輔助函式：針對某個欄位，對 vector<double> 做插值
void linear_interpolate(std::vector<double>& vals) {
    // 找到連續 NaN 區段，以前後已知點線性插值
    // 若開頭或結尾為 NaN，則用最近已知值填補 (類似 forward/backward fill)
    size_t n = vals.size();
    // 先處理整段都 NaN 的極端情況
    bool all_nan = true;
    for (auto v : vals) {
        if (!std::isnan(v)) { all_nan = false; break; }
    }
    if (all_nan) {
        // 全部填 0 或保留 NaN，看需求
        std::fill(vals.begin(), vals.end(), 0.0);
        return;
    }

    // 開頭的連續 NaN，用第一個非 NaN 值填補
    size_t idx = 0;
    while (idx < n && std::isnan(vals[idx])) {
        idx++;
    }
    if (idx > 0 && idx < n) {
        double first_val = vals[idx];
        for (size_t i = 0; i < idx; i++) {
            vals[i] = first_val;
        }
    }

    // 中間區段的 NaN 以前後線性插值
    while (idx < n) {
        // 找到下一個 NaN
        if (!std::isnan(vals[idx])) {
            idx++;
            continue;
        }
        // 起點 left 為 idx-1
        size_t left = idx - 1;
        // 找到連續 NaN 結束處 right
        size_t right = idx;
        while (right < n && std::isnan(vals[right])) {
            right++;
        }
        // 若 right == n，表示後面全部 NaN -> 用前一個值填
        if (right >= n) {
            double last_val = vals[left];
            for (size_t j = left+1; j < n; j++) {
                vals[j] = last_val;
            }
            break;
        }
        // 否則 right 為第一個非 NaN
        double left_val = vals[left];
        double right_val = vals[right];
        size_t gap_len = right - left;
        for (size_t j = left+1; j < right; j++) {
            double ratio = double(j - left) / gap_len;
            vals[j] = left_val + ratio * (right_val - left_val);
        }
        idx = right;
    }
}

// 統一執行線性插值 + forward/backward fill 的邏輯
void interpolate_and_fill(std::vector<StationRecord>& records) {
    // 先把各欄位抽出，分別做
    std::vector<double> temps, hums, precips, winds;
    temps.reserve(records.size());
    hums.reserve(records.size());
    precips.reserve(records.size());
    winds.reserve(records.size());

    for (auto& r : records) {
        temps.push_back(r.air_temperature);
        hums.push_back(r.relative_humidity);
        precips.push_back(r.precipitation);
        winds.push_back(r.wind_speed);
    }

    // 逐欄位線性插值
    linear_interpolate(temps);
    linear_interpolate(hums);
    linear_interpolate(precips);
    linear_interpolate(winds);

    // 寫回
    for (size_t i = 0; i < records.size(); i++) {
        records[i].air_temperature   = temps[i];
        records[i].relative_humidity = hums[i];
        records[i].precipitation     = precips[i];
        records[i].wind_speed        = winds[i];
    }
}

// ----------------------------------------------------------
// 4) 衍生特徵計算 (對應 Python 之 compute_dew_point 等)
// ----------------------------------------------------------
double compute_dew_point(double T, double RH) {
    // 保底避免 RH=0
    if (RH < 1.0) {
        RH = 1.0;
    }
    // 公式: numerator / denominator
    // math.log(RH/100.0) + (17.625 * T)/(243.04 + T)
    // etc.
    double lnRH = std::log(RH / 100.0);
    double part = (17.625 * T) / (243.04 + T);
    double numerator = 243.04 * (lnRH + part);
    double denominator = 17.625 - lnRH - part;
    return numerator / denominator;
}

double compute_apparent_temperature(double T, double V, double RH) {
    // e = (RH/100) * 6.105 * exp((17.27 * T)/(237.7 + T))
    double e = (RH / 100.0) * 6.105 * std::exp((17.27 * T) / (237.7 + T));
    return 1.04 * T + 0.2 * e - 0.65 * V - 2.7;
}

double compute_prob_precipitation(double precip) {
    // 若 precip < 0.1 => 10.0
    // 否則 10 + 70*min(1.0, precip/0.5)
    if (precip < 0.1) {
        return 10.0;
    } else {
        double ratio = std::min(1.0, precip / 0.5);
        return 10.0 + 70.0 * ratio;
    }
}

double compute_comfort_index(double T, double Td) {
    // CI = T - 0.55*(1 - exp((17.269*Td)/(Td+237.3) - (17.269*T)/(T+237.3))) * (T - 14)
    // 以 Python 版為準
    double leftPart = std::exp((17.269 * Td) / (Td + 237.3) - (17.269 * T) / (T + 237.3));
    double factor = (1.0 - leftPart);
    return T - 0.55 * factor * (T - 14.0);
}

// ----------------------------------------------------------
// 5) 產生 "transformed" 資料 (Python transform_data)
// ----------------------------------------------------------
struct TransformedRecord {
    double Temperature;
    double DewPoint;
    double ApparentTemperature;
    double RelativeHumidity;
    double WindSpeed;
    double ProbabilityOfPrecipitation;
    double ComfortIndex;
    std::string date_time;  // 保留原時間，方便後續使用
};

std::vector<TransformedRecord> transform_data(const std::vector<StationRecord>& records) {
    std::vector<TransformedRecord> result;
    result.reserve(records.size());

    for (auto& r : records) {
        TransformedRecord tr;
        tr.Temperature = r.air_temperature;
        tr.RelativeHumidity = r.relative_humidity;
        tr.WindSpeed = r.wind_speed;
        double precip = r.precipitation;

        // dew point
        tr.DewPoint = compute_dew_point(tr.Temperature, tr.RelativeHumidity);
        // apparent temp
        tr.ApparentTemperature = compute_apparent_temperature(tr.Temperature, tr.WindSpeed, tr.RelativeHumidity);
        // prob precipitation
        tr.ProbabilityOfPrecipitation = compute_prob_precipitation(precip);
        // comfort index
        tr.ComfortIndex = compute_comfort_index(tr.Temperature, tr.DewPoint);

        tr.date_time = r.date_time;

        result.push_back(tr);
    }
    return result;
}

// ----------------------------------------------------------
// 6) 將衍生特徵資料拆分成「預測 (predicted)」與「真實 (real)」，並添加隨機誤差
// ----------------------------------------------------------
static std::random_device rd;
static std::mt19937 gen(rd());

// 在一定範圍內產生 ±(2%~5%) 的誤差
double add_random_noise(double value, double min_pct=0.02, double max_pct=0.05) {
    if (std::fabs(value) < 1e-9) {
        // 若原始幾乎為 0，給個小誤差
        std::uniform_real_distribution<> dist_sign(-0.01, 0.01);
        return value + dist_sign(gen);
    }

    std::uniform_real_distribution<> dist(min_pct, max_pct);
    double pct = dist(gen);
    // 50% 機率為負
    std::uniform_int_distribution<> dist_01(0,1);
    if (dist_01(gen) == 1) {
        pct = -pct;
    }
    return value * (1.0 + pct);
}

struct PredictedRecord {
    // 與 TransformedRecord 欄位相同，但數值加誤差
    double Temperature;
    double DewPoint;
    double ApparentTemperature;
    double RelativeHumidity;
    double WindSpeed;
    double ProbabilityOfPrecipitation;
    double ComfortIndex;

    std::string date_time;
};

struct RealRecord {
    double actual_temp;
    double dew_point;
    double apparent_temp;
    double relative_humidity;
    double wind_speed;
    double precipitation;
    double comfort_index;
    bool   actual_rain; // if precipitation > 50 => true
    std::string date_time;
};

// 根據 Python 的 split_predicted_real
void split_predicted_real(
    const std::vector<TransformedRecord>& transformed,
    std::vector<PredictedRecord>& preds,
    std::vector<RealRecord>& reals
) {
    preds.clear();
    reals.clear();
    preds.reserve(transformed.size());
    reals.reserve(transformed.size());

    for (auto& t : transformed) {
        // 1) Predicted
        PredictedRecord p;
        p.Temperature = add_random_noise(t.Temperature);
        p.DewPoint = add_random_noise(t.DewPoint);
        p.ApparentTemperature = add_random_noise(t.ApparentTemperature);
        p.RelativeHumidity = add_random_noise(t.RelativeHumidity);
        p.WindSpeed = add_random_noise(t.WindSpeed);
        p.ProbabilityOfPrecipitation = add_random_noise(t.ProbabilityOfPrecipitation);
        p.ComfortIndex = add_random_noise(t.ComfortIndex);
        p.date_time = t.date_time;
        preds.push_back(p);

        // 2) Real
        RealRecord rrec;
        rrec.actual_temp = add_random_noise(t.Temperature);
        rrec.dew_point = add_random_noise(t.DewPoint);
        rrec.apparent_temp = add_random_noise(t.ApparentTemperature);
        rrec.relative_humidity = add_random_noise(t.RelativeHumidity);
        rrec.wind_speed = add_random_noise(t.WindSpeed);
        // 在 python 裡 precipitation 對應 ProbabilityOfPrecipitation
        rrec.precipitation = add_random_noise(t.ProbabilityOfPrecipitation);
        rrec.comfort_index = add_random_noise(t.ComfortIndex);
        rrec.actual_rain = (t.ProbabilityOfPrecipitation > 50.0);
        rrec.date_time = t.date_time;

        reals.push_back(rrec);
    }
}

// ----------------------------------------------------------
// 7) 最終輸出：與 Python 腳本類似的結構
//    {
//      "input_ids": [...],
//      "predicted_records": [...],
//      "real_records": [...]
//    }
// ----------------------------------------------------------
bool save_combined_data(
    const std::string& output_path,
    const std::vector<PredictedRecord>& preds,
    const std::vector<RealRecord>& reals
) {
    // "input_ids" 對應 Python: "Temperature", "DewPoint", ...
    std::vector<std::string> input_ids = {
        "Temperature",
        "DewPoint",
        "ApparentTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "ProbabilityOfPrecipitation",
        "ComfortIndex"
    };

    // 建立 JSON 結構
    json j;
    j["input_ids"] = input_ids;

    // predicted_records
    {
        json arr = json::array();
        for (auto& p : preds) {
            json pj;
            pj["Temperature"] = p.Temperature;
            pj["DewPoint"] = p.DewPoint;
            pj["ApparentTemperature"] = p.ApparentTemperature;
            pj["RelativeHumidity"] = p.RelativeHumidity;
            pj["WindSpeed"] = p.WindSpeed;
            pj["ProbabilityOfPrecipitation"] = p.ProbabilityOfPrecipitation;
            pj["ComfortIndex"] = p.ComfortIndex;
            pj["date_time"] = p.date_time;
            arr.push_back(pj);
        }
        j["predicted_records"] = arr;
    }

    // real_records
    {
        json arr = json::array();
        for (auto& r : reals) {
            json rj;
            rj["actual_temp"] = r.actual_temp;
            rj["dew_point"] = r.dew_point;
            rj["apparent_temp"] = r.apparent_temp;
            rj["relative_humidity"] = r.relative_humidity;
            rj["wind_speed"] = r.wind_speed;
            rj["precipitation"] = r.precipitation;
            rj["comfort_index"] = r.comfort_index;
            rj["actual_rain"] = r.actual_rain;
            rj["date_time"] = r.date_time;
            arr.push_back(rj);
        }
        j["real_records"] = arr;
    }

    // 開啟檔案
    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "無法開啟輸出檔案: " << output_path << std::endl;
        return false;
    }

    // 設定 UTF-8 locale
    // try {
    //     // Linux/macOS 通常可用 "en_US.UTF-8" 或 "zh_TW.UTF-8"
    //     // Windows 可嘗試 ".UTF8" 或 ".65001" 等
    //     std::locale utf8Locale("en_US.UTF-8");
    //     ofs.imbue(utf8Locale);
    // } catch (const std::runtime_error& e) {
    //     std::cerr << "警告：無法載入指定 locale (en_US.UTF-8)，系統可能不支援該 locale。\n";
    //     // 這裡可忽略錯誤，或改用預設 locale，或使用其他 fallback
    // }

    // 寫檔 (nlohmann/json 寫入的字串若已是 UTF-8，就會直接輸出 UTF-8)
    ofs << std::setw(4) << j << std::endl;
    ofs.close();
    return true;
}


// ----------------------------------------------------------
// 8) 主程式：讀取 summary.json -> 依測站 ID 處理 -> 清洗/補值 -> 衍生特徵 -> 預測與真實 -> 輸出
// ----------------------------------------------------------
int main() {
    // (A) 讀取 summary.json
    std::string summary_path = "../../data/Data_history_clean/summary.json";
    std::ifstream summary_ifs(summary_path);
    if (!summary_ifs.is_open()) {
        std::cerr << "無法開啟 summary.json\n";
        return 1;
    }
    json summary_j;
    try {
        summary_ifs >> summary_j;
    } catch (...) {
        std::cerr << "解析 summary.json 失敗\n";
        return 1;
    }

    // 檢查 "stations" 結構
    if (!summary_j.contains("stations") || !summary_j["stations"].is_object()) {
        std::cerr << "summary.json 缺少 stations 結構\n";
        return 1;
    }
    auto stations_obj = summary_j["stations"];

    // 輸出路徑
    fs::path output_dir = "../../data/Data_history_final";
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // (B) 逐測站檢查
    for (auto& [station_id, meta] : stations_obj.items()) {
        // meta 應包含 {"count", "id_min", "id_max", "status"}
        if (!meta.contains("count")) {
            continue;
        }
        long long count = meta["count"].get<long long>();

        // 若資料總數 < 10000 則跳過
        if (count < 10000) {
            std::cout << "測站 " << station_id << " 筆數小於一萬，略過\n";
            continue;
        }

        // (C) 讀取該測站的 JSON
        // 假設檔名與 station_id 相同，例如 "C0X100.json"
        fs::path station_file = fs::path("../../data/Data_history_clean/") / (station_id + ".json");
        // 或者如果原檔案不在當前資料夾，請依實際路徑調整
        if (!fs::exists(station_file)) {
            std::cout << "找不到測站檔案: " << station_file << "，略過\n";
            continue;
        }

        std::vector<StationRecord> records;
        bool ok = load_station_data(station_file.string(), records);
        if (!ok || records.empty()) {
            std::cout << "讀取失敗或無資料，測站: " << station_id << " 略過\n";
            continue;
        }
        std::cout << "測站 " << station_id << " 資料筆數: " << records.size() << std::endl;

        // (D) 清洗與補值
        replace_missing_with_nan(records);
        interpolate_and_fill(records);
        
        // (E) [可選] 排序 date_time (若有需要和 Python 一樣先依時間排序)
        // 這裡僅示範簡單字串比較；嚴謹可使用時間解析再排序
        std::sort(records.begin(), records.end(), [](auto& a, auto& b){
            return a.date_time < b.date_time;
        });

        // (F) 產生衍生特徵
        auto transformed = transform_data(records);

        // (G) 分割成 predicted & real
        std::vector<PredictedRecord> preds;
        std::vector<RealRecord> reals;
        split_predicted_real(transformed, preds, reals);

        // (H) 最終輸出: station_id.json => ./data/Data_history_final/station_id.json
        fs::path out_path = output_dir / (station_id + ".json");
        if (save_combined_data(out_path.string(), preds, reals)) {
            std::cout << "已完成測站 " << station_id << "，輸出檔案: " << out_path.string() << std::endl;
        }
        
        // (I) 清理該測站的資料暫存 (釋放記憶體)
        records.clear();
        transformed.clear();
        preds.clear();
        reals.clear();
        // 當離開此區塊後，相關 vector 也不再佔用記憶體
    }

    std::cout << "=== 所有測站處理完畢 ===\n";
    return 0;
}
