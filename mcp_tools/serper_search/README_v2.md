# Serper Search MCP Server v2.0

全新重構的 Serper API MCP 伺服器，代碼更乾淨、功能更完整。

## 🆕 v2.0 新功能

### 1. **新增 4 個搜尋工具**
- ✅ `search_news` - Google 新聞搜尋（支援時間過濾）
- ✅ `search_images` - 圖片搜尋
- ✅ `search_videos` - 影片搜尋
- ✅ `autocomplete` - 搜尋建議

### 2. **增強的參數支援**
所有搜尋工具現在支援：
- `tbs` - 時間過濾（`qdr:h`=1小時, `qdr:d`=1天, `qdr:w`=1週, `qdr:m`=1月, `qdr:y`=1年）
- `hl` - 語言代碼（`en`, `es`, `fr`, `zh-TW` 等）
- `autocorrect` - 自動拼寫糾正（預設啟用）
- `page` - 分頁支援

### 3. **修復的 Bug**
- ✅ `scrape_webpage` 現在正確返回 `str` 而非錯誤的 `dict`
- ✅ 改進的錯誤處理（401, 429, 500 錯誤）
- ✅ 更清晰的錯誤訊息

### 4. **代碼改進**
- ✅ Pydantic Settings 配置管理
- ✅ 所有請求參數使用 Pydantic Models 驗證
- ✅ `SerperClient` 類封裝 API 邏輯
- ✅ 延遲載入 tokenizer（節省啟動時間）
- ✅ 無全域變數（測試友好）

### 5. **靈活的 Transport**
同時支援 `stdio` 和 `http` transport：
```bash
# STDIO (預設，用於 Claude Desktop)
python server_v2.py

# HTTP (用於網路服務)
python server_v2.py --transport http --port 8000

# 自訂日誌等級
python server_v2.py --log-level DEBUG
```

---

## 📋 完整工具列表

### 1. `search` - 網頁搜尋
```python
{
    "q": "artificial intelligence",
    "num": 10,
    "gl": "us",
    "hl": "en",
    "page": 1,
    "tbs": "qdr:w"  # 最近一週
}
```

### 2. `search_news` - 新聞搜尋
```python
{
    "q": "AI breakthrough",
    "num": 20,
    "tbs": "qdr:d"  # 最近一天
}
```

### 3. `search_images` - 圖片搜尋
```python
{
    "q": "nature photography",
    "num": 50
}
```

### 4. `search_videos` - 影片搜尋
```python
{
    "q": "python tutorial",
    "num": 10
}
```

### 5. `search_places` - 地點搜尋
```python
{
    "q": "coffee shop near me",
    "gl": "us"
}
```

### 6. `autocomplete` - 搜尋建議
```python
{
    "q": "how to"
}
```

### 7. `scrape_webpage` - 網頁抓取
```python
{
    "url": "https://example.com",
    "include_markdown": false
}
```
**返回**: 文字內容（自動截斷到 4000 tokens）

---

## ⚙️ 配置

### 環境變數

創建 `.env` 文件或設置環境變數：

```bash
# 必需
SERPER_API_KEY=your_api_key_here

# 可選
AIOHTTP_TIMEOUT=15              # HTTP 請求超時（秒）
MAX_WEB_TOKENS=4000            # 網頁抓取最大 tokens
TOKENIZER_PATH=Qwen/Qwen2.5-0.5B  # Tokenizer 模型路徑
ENABLE_TOKENIZER=true          # 啟用 tokenizer（false 則使用簡單截斷）
```

### 命令行參數

```bash
python server_v2.py [options]

Options:
  --transport {stdio,http}   Transport protocol (預設: stdio)
  --host HOST               HTTP 主機地址 (預設: 127.0.0.1)
  --port PORT               HTTP 端口 (預設: 8000)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                           日誌等級 (預設: INFO)
```

---

## 🚀 使用範例

### 基本網頁搜尋
```python
await client.call_tool("search", {
    "q": "FastMCP tutorial",
    "num": 5
})
```

### 搜尋最新新聞
```python
await client.call_tool("search_news", {
    "q": "OpenAI",
    "tbs": "qdr:h",  # 最近一小時
    "num": 10
})
```

### 獲取搜尋建議
```python
await client.call_tool("autocomplete", {
    "q": "best programming"
})
# 返回: ["best programming languages", "best programming laptop", ...]
```

### 抓取網頁內容
```python
result = await client.call_tool("scrape_webpage", {
    "url": "https://news.ycombinator.com",
    "include_markdown": true
})
# 返回 Markdown 格式的文字內容（已截斷）
```

---

## 🔧 依賴項

```bash
pip install fastmcp aiohttp certifi pydantic pydantic-settings

# 可選（用於網頁抓取）
pip install transformers
```

或使用 `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📊 v1 vs v2 比較

| 功能 | v1 (server.py) | v2 (server_v2.py) |
|------|----------------|-------------------|
| 工具數量 | 3 | 7 |
| Transport 支援 | stdio, http, sse | stdio, http |
| 配置管理 | 環境變數 + 全域變數 | Pydantic Settings |
| 錯誤處理 | 基本 | ToolError + HTTP 狀態碼 |
| 參數驗證 | Pydantic (部分) | 完整 Pydantic Models |
| Tokenizer | 啟動時載入 | 延遲載入（按需） |
| 代碼組織 | 平面結構 | 模組化（Config/Models/Client/Tools） |
| scrape_webpage bug | ❌ 返回類型錯誤 | ✅ 已修復 |
| 測試友好 | ⚠️ 全域變數 | ✅ 依賴注入 |

---

## 🧪 測試

### 測試 STDIO Transport
```bash
export SERPER_API_KEY=your_key
python server_v2.py --log-level DEBUG
```

### 測試 HTTP Transport
```bash
# 終端 1：啟動伺服器
python server_v2.py --transport http --port 8000

# 終端 2：測試
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {"q": "test", "num": 5}
    },
    "id": 1
  }'
```

---

## 🛠️ 疑難排解

### 問題：`SERPER_API_KEY` 未設置
**解決**：
```bash
export SERPER_API_KEY=your_api_key
# 或創建 .env 文件
```

### 問題：Tokenizer 下載失敗
**解決**：
```bash
# 方案 1：停用 tokenizer（使用簡單截斷）
export ENABLE_TOKENIZER=false

# 方案 2：使用更小的模型
export TOKENIZER_PATH=Qwen/Qwen2.5-0.5B
```

### 問題：HTTP transport 連接失敗
**解決**：
- 檢查防火牆設置
- 確認端口未被佔用
- 嘗試使用 `--host 0.0.0.0` 允許外部連接

---

## 📝 未來改進

- [ ] 添加重試邏輯（使用 `aiohttp-retry`）
- [ ] 響應快取（減少 API 調用）
- [ ] 更多 Serper endpoints（scholar, patents, shopping）
- [ ] 速率限制中間件
- [ ] 完整的單元測試覆蓋

---

## 📄 授權

與原專案相同。

## 🔗 相關連結

- [Serper API 文檔](https://serper.dev)
- [FastMCP 文檔](https://gofastmcp.com)
- [MCP 協議規範](https://modelcontextprotocol.io)
