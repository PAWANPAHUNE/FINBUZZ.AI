
# FINBUZZ.AI 🧠💸

FINBUZZ.AI is an **Agentic AI-powered Personal Financial Assistant** that brings together a diverse set of financial tools—traditionally scattered across multiple platforms—into one **unified, intelligent system**.

Built with a **multi-agent architecture**, large language models, retrieval-augmented generation (RAG), and conversational UI, FINBUZZ.AI acts as your **personal investment and finance advisor** through **chat** or **voice**.

---

## 🚀 Key Features

### 🔧 Personalized Financial Tools
- **Voice Interaction, JSON/Text Data Input & Report Generator**  
  → Upload financial data or speak your queries to generate personalized PDF reports.

- **Net Worth Tracker & Projection Engine**  
  → Monitor your finances, simulate future milestones, and get a 0–100 financial health score.

- **SIP & Mutual Fund Analyzer**  
  → Analyze fund performance, detect overexposure, and identify weak investments.

- **Loan & Debt Optimization Advisor**  
  → Assess EMI affordability, restructure debt, and minimize interest.

- **Green Investing & Behavioral Bias Alerts**  
  → Promote ESG investing and detect risky patterns like panic-selling.

- **Risk Appetite Profiler & Goal Planner**  
  → Tailor investment plans based on risk tolerance and life goals.

- **Interactive Visual Guidance**  
  → Get guidance with dynamic charts, diagrams, and conversational flows.

- **“What-If” Engine & Tax Optimizer**  
  → Simulate scenarios and maximize tax savings within legal limits.

- **EPF/PPF Optimizer & Periodic Reports**  
  → Optimize retirement contributions and receive scheduled reports.

---

### 📊 Financial Analytics & Investment Tools
- **Forecaster, Macro Dashboard, Sector Picker**  
  → Analyze macroeconomic trends and discover promising sectors.

- **Smart Stock Screener, Sentiment Engine, Valuation Comparator**  
  → Screen strong stocks, evaluate social/news sentiment, and benchmark valuations.

- **Portfolio Builder, Rebalancer & Risk Heatmaps**  
  → Create optimized portfolios and visualize risk exposure.

- **IPO Advisor, Event Detector & Dividend Optimizer**  
  → Identify lucrative IPOs, detect corporate actions, and enhance dividend yield.

- **Thematic & Sector-Rotation Portfolios**  
  → Build theme-based or rotating sector portfolios with ease.

- **Natural Language Screening + AI Fund Manager**  
  → Screen stocks using natural language and receive real-time advice from an intelligent agent.

---

## 🛠️ Tech Stack

- 🧠 **LLM-Orchestrated Multi-Agent System**
- 🔍 **Retrieval-Augmented Generation (RAG)**
- 📦 `LangChain`, `OpenAI`, `Pandas`, `Matplotlib`, `FPDF`
- 📈 Financial APIs and datasets
- 🎤 **Speech Recognition** + **Text-to-Speech** Integration

---

## ⚙️ How to Run Locally

> 💡 Make sure you have Python 3.8+ installed

### Step-by-Step Setup:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/finbuzz-ai.git
   cd finbuzz-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open in your IDE**
   - Open the project in **VS Code** or **any Python-compatible IDE**.

4. **Run the main agent**
   ```bash
   python agent5.py
   ```

5. **Start using FINBUZZ.AI**
   - Interact with the assistant via **chat** or **voice commands**
   - Upload a `.json` or `.txt` financial file for personalized insights

---

## 📎 Example JSON Input Format

```json
{
  "name": "John Doe",
  "monthly_income": 100000,
  "expenses": 40000,
  "savings": 200000,
  "investments": [
    {"type": "mutual_fund", "amount": 100000, "category": "equity"},
    {"type": "fixed_deposit", "amount": 50000}
  ],
  "loans": [
    {"type": "home_loan", "emi": 20000, "interest": 7.5}
  ],
  "goals": [
    {"goal": "retirement", "target_year": 2045, "amount": 8000000}
  ]
}
```

---

## 📌 Future Enhancements

- Web interface (React + Flask)
- Integration with real-time market APIs
- Auto-import financial data from bank/email APIs
- Personalized financial nudges via WhatsApp or Telegram

---

## 👨‍💻 Author

**Pawan Pahune**  
`Developer | Financial Analyst | Agentic AI Enthusiast`
