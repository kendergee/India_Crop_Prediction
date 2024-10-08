
# 印度稻米產量預測程式

## 簡介

本程式旨在預測印度各區域和州的稻米單位面積產量（Yield）。使用者可以選擇區域和州，並輸入相關參數，如降雨量、季節和總產量，程式將根據這些數據預測稻米的單位面積產量，幫助農民、農業管理者以及政策制定者做出更明智的決策。

## 功能特點

- **區域與州的選擇**：使用者可以從印度的不同區域中選擇特定的州，以獲取更精確的預測結果。
- **多參數輸入**：程式允許使用者輸入多個關鍵參數，包括降雨量、季節等，以進行更為精確的產量預測。
- **單位面積產量（Yield）預測**：根據輸入的參數，程式會自動計算並返回預測的單位面積產量。

## 使用步驟

1. **啟動程式**：
   - 確保程式文件與所需的模型和資料檔案位於同一個目錄下。
   - 打開終端或命令提示符，導航到程式所在的目錄，輸入 `python Rice_yield_predict.py` 來啟動程式。

2. **選擇區域和州**：
   - 程式啟動後，首先會提示你選擇想要進行預測的區域。
   - 從下拉選單選擇區域、州、季節

3. **輸入相關參數**：
   - **年份**：程式會提示你輸入的年份，例如 `2020`。
   - **種植面積**：程式會提示你輸入當前季節的種植面積（公頃），例如 `800`。
   - **降雨量**：程式會提示你輸入當前季節的年降雨量（毫米），例如 `800`。
   - **肥料使用量**：根據提示輸入使用的肥料總量（單位：公斤）。
   - **農藥使用量**：根據提示輸入使用的農藥總量（單位：公升）。

5. **預測結果**：
   - 所有參數輸入完畢後，程式將會計算並顯示預測的單位面積產量（Yield）以及產出圖表（如果資料夠的話）。
   - 預測結果會顯示在終端上，你可以記錄或將結果保存至外部文件。
