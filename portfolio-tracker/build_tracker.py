"""
Builds portfolio_tracker.xlsx, a Google Sheets portfolio tracker.

The workbook is meant to be opened in Google Sheets (upload it to Drive,
or File > Import). All market data comes from GOOGLEFINANCE, so prices,
history and every metric update on their own. It will NOT calculate in
Excel, since GOOGLEFINANCE is a Sheets-only function.

Tabs:
    Dashboard   KPIs (return, alpha, beta, volatility, Sharpe, drawdown),
                sector allocation and charts
    Holdings    current positions with live prices, P/L and weights
    Trades      buy/sell log plus the thesis behind each decision (inputs)
    Settings    start date, starting cash, benchmark, risk-free rate (inputs)
    History     daily portfolio vs benchmark values, returns and drawdowns
    Prices      raw daily closes pulled by GOOGLEFINANCE

Usage:
    python build_tracker.py            # writes portfolio_tracker.xlsx
"""

import datetime as dt

from openpyxl import Workbook
from openpyxl.chart import LineChart, PieChart, AreaChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

OUT = "portfolio_tracker.xlsx"

SLOTS = 10            # ticker slots in Holdings / History / Prices
TRADE_ROWS = 200      # rows available in the trade log
HIST_ROWS = 1000      # trading days of history the formulas cover
STYLED_TRADE_ROWS = 40  # trade rows pre-formatted as inputs (keeps the file small)

# Example mock portfolio. Replace with your own picks.
START_DATE = dt.date(2026, 1, 2)
START_CASH = 100_000
BENCHMARK = "SPY"     # e.g. XIC for the S&P/TSX (keep everything in one currency)
RISK_FREE = 0.04

HOLDINGS = [
    ("AAPL", "Information Technology"),
    ("MSFT", "Information Technology"),
    ("JPM", "Financials"),
    ("XOM", "Energy"),
    ("PG", "Consumer Staples"),
    ("UNH", "Health Care"),
    ("CAT", "Industrials"),
    ("NEE", "Utilities"),
]

# (date, ticker, action, shares, thesis, catalyst, key risk, exit trigger)
TRADES = [
    (START_DATE, "AAPL", "BUY", 40,
     "Example: cash-generative mega-cap; services mix supports margins",
     "Next earnings / product cycle", "Hardware demand slowdown",
     "Sell if services growth stalls two quarters in a row"),
    (START_DATE, "MSFT", "BUY", 20,
     "Example: cloud + enterprise software compounder",
     "Cloud growth at earnings", "AI capex not earning a return",
     "Trim if weight exceeds 15%"),
    (START_DATE, "JPM", "BUY", 32,
     "Example: best-in-class bank, benefits from higher-for-longer rates",
     "Net interest income guidance", "Credit losses rise in a slowdown",
     "Review if rate cuts compress NII"),
    (START_DATE, "XOM", "BUY", 85,
     "Example: energy exposure as an inflation hedge",
     "Oil price / OPEC decisions", "Oil price collapse",
     "Cut if position falls 15% below cost"),
    (START_DATE, "PG", "BUY", 65,
     "Example: low-beta defensive ballast",
     "Pricing power in quarterly results", "Volume declines from price hikes",
     "Hold as long as it lowers portfolio beta"),
    (START_DATE, "UNH", "BUY", 30,
     "Example: healthcare diversification",
     "Medical cost trend updates", "Regulatory / reimbursement changes",
     "Exit on a guidance cut"),
    (START_DATE, "CAT", "BUY", 18,
     "Example: cyclical play on infrastructure spending",
     "Order backlog at earnings", "Industrial recession",
     "Trim if backlog shrinks"),
    (START_DATE, "NEE", "BUY", 130,
     "Example: utilities exposure and rate sensitivity",
     "Rate cuts", "Rates stay high",
     "Review if 10y yield rises sharply"),
    (dt.date(2026, 6, 1), "XOM", "SELL", 25,
     "Example: trim to rebalance energy weight",
     "", "", ""),
]

# ---------------------------------------------------------------- styles
NAVY = "1F3864"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
KPI_FILL = PatternFill("solid", fgColor="EEF3FA")
TITLE_FONT = Font(size=16, bold=True, color=NAVY)
HEADER_FONT = Font(bold=True, color="FFFFFF")
INPUT_FONT = Font(color="0000FF")
NOTE_FONT = Font(italic=True, color="666666", size=9)
BOLD = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)

USD = '"$"#,##0.00'
USD0 = '"$"#,##0'
PCT = "0.00%"
NUM2 = "0.00"
DATE = "yyyy-mm-dd"


def header(ws, row, labels, col=1):
    for i, label in enumerate(labels):
        c = ws.cell(row=row, column=col + i, value=label)
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def title(ws, text, note=None):
    ws["A1"] = text
    ws["A1"].font = TITLE_FONT
    if note:
        ws["A2"] = note
        ws["A2"].font = NOTE_FONT


def mark_input(cell):
    cell.fill = INPUT_FILL
    cell.font = INPUT_FONT


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


def col(n):
    from openpyxl.utils import get_column_letter
    return get_column_letter(n)


# Cell references shared across tabs
START = "Settings!$B$5"
CASH0 = "Settings!$B$6"
BENCH = "Settings!$B$7"
RF = "Settings!$B$8"
T_LAST = 4 + TRADE_ROWS
T_DATE = f"Trades!$A$5:$A${T_LAST}"
T_TICK = f"Trades!$B$5:$B${T_LAST}"
T_ACT = f"Trades!$C$5:$C${T_LAST}"
T_SH = f"Trades!$D$5:$D${T_LAST}"
T_PX = f"Trades!$E$5:$E${T_LAST}"
T_SIGNED = f"Trades!$G$5:$G${T_LAST}"
T_FLOW = f"Trades!$H$5:$H${T_LAST}"
H_FIRST, H_LAST = 4, 3 + HIST_ROWS     # History / Prices data rows
HOLD_FIRST = 5
HOLD_LAST = HOLD_FIRST + SLOTS - 1


def build():
    wb = Workbook()
    dash = wb.active
    dash.title = "Dashboard"
    hold = wb.create_sheet("Holdings")
    trades = wb.create_sheet("Trades")
    settings = wb.create_sheet("Settings")
    hist = wb.create_sheet("History")
    prices = wb.create_sheet("Prices")

    # ------------------------------------------------------------ Settings
    title(settings, "Settings",
          "Yellow cells with blue text are inputs. Everything else is calculated.")
    rows = [
        ("Portfolio name", "Mock Equity Portfolio", None),
        ("Start date (inception)", START_DATE, DATE),
        ("Starting cash", START_CASH, USD0),
        ("Benchmark ticker", BENCHMARK, None),
        ("Risk-free rate (annual)", RISK_FREE, PCT),
    ]
    for i, (label, value, fmt) in enumerate(rows, start=4):
        settings.cell(row=i, column=1, value=label).font = BOLD
        c = settings.cell(row=i, column=2, value=value)
        mark_input(c)
        if fmt:
            c.number_format = fmt
    settings["A11"] = "Notes"
    settings["A11"].font = BOLD
    notes = [
        "Prices come from GOOGLEFINANCE and can be delayed up to 20 minutes.",
        "Benchmark: SPY tracks the S&P 500. For the S&P/TSX use XIC (TSE:XIC), "
        "and keep holdings in the same currency so returns are comparable.",
        "Returns are price returns; dividends are not included.",
        "Every ticker you trade must also be listed in a Holdings slot.",
        "History covers up to 1,000 trading days from the start date.",
    ]
    for i, n in enumerate(notes, start=12):
        settings.cell(row=i, column=1, value="• " + n)
    widths(settings, {"A": 28, "B": 26})

    # -------------------------------------------------------------- Trades
    title(trades, "Trade Log & Investment Thesis",
          "Log every buy/sell with the reasoning behind it. Price defaults to that "
          "day's close via GOOGLEFINANCE; overwrite it with your actual fill if you have one.")
    header(trades, 4, ["Date", "Ticker", "Action", "Shares", "Price", "Fees",
                       "Signed shares", "Cash flow", "Thesis", "Catalyst",
                       "Key risk", "Exit / review trigger"])
    for r, t in enumerate(TRADES, start=5):
        date, tick, action, shares, thesis, catalyst, risk, exit_ = t
        trades.cell(row=r, column=1, value=date)
        trades.cell(row=r, column=2, value=tick)
        trades.cell(row=r, column=3, value=action)
        trades.cell(row=r, column=4, value=shares)
        trades.cell(row=r, column=5,
                    value=f'=IFERROR(INDEX(GOOGLEFINANCE(B{r},"close",A{r}),2,2),"")')
        trades.cell(row=r, column=6, value=0)
        for c, v in zip(range(9, 13), (thesis, catalyst, risk, exit_)):
            trades.cell(row=r, column=c, value=v or None)
    for r in range(5, 5 + STYLED_TRADE_ROWS):
        for c in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12):
            mark_input(trades.cell(row=r, column=c))
        trades.cell(row=r, column=1).number_format = DATE
        trades.cell(row=r, column=5).number_format = USD
        trades.cell(row=r, column=6).number_format = USD
        trades.cell(row=r, column=7).number_format = "#,##0"
        trades.cell(row=r, column=8).number_format = USD
        for c in range(9, 13):
            trades.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
    trades["G5"] = (f'=ARRAYFORMULA(IF(B5:B{T_LAST}="",,'
                    f'IF(C5:C{T_LAST}="SELL",-D5:D{T_LAST},D5:D{T_LAST})))')
    trades["H5"] = (f'=ARRAYFORMULA(IF(B5:B{T_LAST}="",,'
                    f'-G5:G{T_LAST}*E5:E{T_LAST}-F5:F{T_LAST}))')
    dv = DataValidation(type="list", formula1='"BUY,SELL"', allow_blank=True)
    trades.add_data_validation(dv)
    dv.add(f"C5:C{T_LAST}")
    trades.freeze_panes = "C5"
    widths(trades, {"A": 12, "B": 9, "C": 8, "D": 9, "E": 11, "F": 8, "G": 9,
                    "H": 13, "I": 46, "J": 28, "K": 28, "L": 34})

    # ------------------------------------------------------------ Holdings
    title(hold, "Holdings",
          "Enter a ticker and sector in each slot (yellow). Shares and cost come "
          "from the Trades tab; prices are live.")
    header(hold, 4, ["Ticker", "Name", "Sector", "Shares", "Avg cost",
                     "Live price", "Day chg", "Market value", "Weight",
                     "Cost basis", "Unrealized P/L", "Return", "Beta"])
    total_row = HOLD_LAST + 1
    cash_row = total_row + 1
    port_row = cash_row + 1
    for i in range(SLOTS):
        r = HOLD_FIRST + i
        if i < len(HOLDINGS):
            hold.cell(row=r, column=1, value=HOLDINGS[i][0])
            hold.cell(row=r, column=3, value=HOLDINGS[i][1])
        mark_input(hold.cell(row=r, column=1))
        mark_input(hold.cell(row=r, column=3))
        f = {
            2: f'=IF(A{r}="","",IFERROR(GOOGLEFINANCE(A{r},"name"),""))',
            4: f'=IF(A{r}="","",SUMIFS({T_SIGNED},{T_TICK},A{r}))',
            5: (f'=IF(A{r}="","",IFERROR(SUMPRODUCT(({T_TICK}=A{r})*({T_ACT}="BUY"),'
                f'{T_SH},{T_PX})/SUMIFS({T_SH},{T_TICK},A{r},{T_ACT},"BUY"),""))'),
            6: f'=IF(A{r}="","",IFERROR(GOOGLEFINANCE(A{r}),""))',
            7: f'=IF(A{r}="","",IFERROR(GOOGLEFINANCE(A{r},"changepct")/100,""))',
            8: f'=IF(A{r}="","",IFERROR(D{r}*F{r},0))',
            9: f'=IF(A{r}="","",IFERROR(H{r}/$H${port_row},""))',
            10: f'=IF(A{r}="","",IFERROR(D{r}*E{r},0))',
            11: f'=IF(A{r}="","",H{r}-J{r})',
            12: f'=IF(A{r}="","",IFERROR(K{r}/J{r},""))',
            13: f'=IF(A{r}="","",IFERROR(GOOGLEFINANCE(A{r},"beta"),""))',
        }
        for c, formula in f.items():
            hold.cell(row=r, column=c, value=formula)
        for c, fmt in {4: "#,##0", 5: USD, 6: USD, 7: PCT, 8: USD, 9: PCT,
                       10: USD, 11: USD, 12: PCT, 13: NUM2}.items():
            hold.cell(row=r, column=c).number_format = fmt
    rng = lambda c: f"{c}{HOLD_FIRST}:{c}{HOLD_LAST}"
    hold.cell(row=total_row, column=1, value="Total equities").font = BOLD
    hold.cell(row=total_row, column=8, value=f"=SUM({rng('H')})")
    hold.cell(row=total_row, column=9, value=f"=IFERROR(H{total_row}/H{port_row},0)")
    hold.cell(row=total_row, column=10, value=f"=SUM({rng('J')})")
    hold.cell(row=total_row, column=11, value=f"=H{total_row}-J{total_row}")
    hold.cell(row=total_row, column=12, value=f"=IFERROR(K{total_row}/J{total_row},0)")
    hold.cell(row=cash_row, column=1, value="Cash").font = BOLD
    hold.cell(row=cash_row, column=8, value=f"={CASH0}+SUM({T_FLOW})")
    hold.cell(row=cash_row, column=9, value=f"=IFERROR(H{cash_row}/H{port_row},0)")
    hold.cell(row=port_row, column=1, value="Total portfolio").font = BOLD
    hold.cell(row=port_row, column=8, value=f"=H{total_row}+H{cash_row}")
    hold.cell(row=port_row, column=9, value=f"=H{total_row}/H{port_row}+I{cash_row}")
    for r in (total_row, cash_row, port_row):
        for c in range(1, 14):
            hold.cell(row=r, column=c).border = Border(top=THIN)
        for c, fmt in {8: USD, 9: PCT, 10: USD, 11: USD, 12: PCT}.items():
            hold.cell(row=r, column=c).number_format = fmt
            hold.cell(row=r, column=c).font = BOLD
    hold.freeze_panes = "B5"
    widths(hold, {"A": 15, "B": 30, "C": 24, "D": 9, "E": 11, "F": 11, "G": 9,
                  "H": 14, "I": 9, "J": 14, "K": 14, "L": 9, "M": 7})

    # -------------------------------------------------------------- Prices
    title(prices, "Raw Prices",
          "Daily closes from GOOGLEFINANCE, starting at the Settings start date. "
          "Don't type here.")
    prices["A3"] = f'=GOOGLEFINANCE({BENCH},"close",{START},TODAY(),"DAILY")'
    for i in range(SLOTS):
        c = col(3 + 2 * i)
        t = f"Holdings!$A${HOLD_FIRST + i}"
        prices[f"{c}3"] = f'=IFERROR(GOOGLEFINANCE({t},"close",{START},TODAY(),"DAILY"),"")'
    for i in range(SLOTS + 1):
        prices.column_dimensions[col(1 + 2 * i)].number_format = DATE
    widths(prices, {col(c): 11 for c in range(1, 3 + 2 * SLOTS)})

    # ------------------------------------------------------------- History
    title(hist, "Daily History",
          "Portfolio value is rebuilt each day from the trade log and closing prices. "
          "Don't type here.")
    labels = ["Date", "Benchmark close"]
    labels += [f"=IF(Holdings!A{HOLD_FIRST + i}=\"\",\"Slot {i + 1}\","
               f"Holdings!A{HOLD_FIRST + i}&\" value\")" for i in range(SLOTS)]
    labels += ["Equities", "Cash", "Portfolio value", "Portfolio (base 100)",
               "Benchmark (base 100)", "Portfolio daily ret", "Benchmark daily ret",
               "Portfolio peak", "Portfolio drawdown", "Benchmark peak",
               "Benchmark drawdown"]
    header(hist, 3, labels)
    R = lambda c, first=H_FIRST, last=H_LAST: f"{c}{first}:{c}{last}"
    hist["A4"] = f"=ARRAYFORMULA(IF(Prices!{R('A')}=\"\",,Prices!{R('A')}))"
    hist["B4"] = f"=ARRAYFORMULA(IF(Prices!{R('A')}=\"\",,Prices!{R('B')}))"
    for i in range(SLOTS):
        c = col(3 + i)
        t = f"Holdings!$A${HOLD_FIRST + i}"
        pd_, pc = col(3 + 2 * i), col(4 + 2 * i)
        hist[f"{c}4"] = (
            f'=IF({t}="",,MAP({R("$A")},LAMBDA(d,IF(d="",,'
            f'SUMIFS({T_SIGNED},{T_TICK},{t},{T_DATE},"<="&d)*'
            f'IFERROR(XLOOKUP(d,Prices!${pd_}${H_FIRST}:${pd_}${H_LAST},'
            f'Prices!${pc}${H_FIRST}:${pc}${H_LAST},0,-1),0)))))')
    first_val, last_val = col(3), col(2 + SLOTS)
    c_eq, c_cash, c_port, c_pidx, c_bidx, c_pret, c_bret, c_ppk, c_pdd, c_bpk, c_bdd = (
        col(3 + SLOTS + k) for k in range(11))
    hist[f"{c_eq}4"] = (f'=ARRAYFORMULA(IF({R("A")}="",,MMULT(IFERROR('
                        f'{first_val}{H_FIRST}:{last_val}{H_LAST}*1,0),SEQUENCE({SLOTS},1,1,0))))')
    hist[f"{c_cash}4"] = (f'=MAP({R("A")},LAMBDA(d,IF(d="",,'
                          f'{CASH0}+SUMIFS({T_FLOW},{T_DATE},"<="&d))))')
    hist[f"{c_port}4"] = f'=ARRAYFORMULA(IF({R("A")}="",,{R(c_eq)}+{R(c_cash)}))'
    hist[f"{c_pidx}4"] = f'=ARRAYFORMULA(IF({R("A")}="",,{R(c_port)}/${c_port}${H_FIRST}*100))'
    hist[f"{c_bidx}4"] = f'=ARRAYFORMULA(IF({R("A")}="",,{R("B")}/$B${H_FIRST}*100))'
    nxt = H_FIRST + 1
    hist[f"{c_pret}5"] = (f'=ARRAYFORMULA(IF({R("A", nxt)}="",,'
                          f'{R(c_port, nxt)}/{R(c_port, H_FIRST, H_LAST - 1)}-1))')
    hist[f"{c_bret}5"] = (f'=ARRAYFORMULA(IF({R("A", nxt)}="",,'
                          f'{R("B", nxt)}/{R("B", H_FIRST, H_LAST - 1)}-1))')
    hist[f"{c_ppk}4"] = f'=SCAN(0,{R(c_port)},LAMBDA(a,v,IF(v="",a,MAX(a,v))))'
    hist[f"{c_pdd}4"] = f'=ARRAYFORMULA(IF({R("A")}="",,{R(c_port)}/{R(c_ppk)}-1))'
    hist[f"{c_bpk}4"] = f'=SCAN(0,{R("B")},LAMBDA(a,v,IF(v="",a,MAX(a,v))))'
    hist[f"{c_bdd}4"] = f'=ARRAYFORMULA(IF({R("A")}="",,{R("B")}/{R(c_bpk)}-1))'
    fmts = {"A": DATE, "B": USD}
    fmts.update({col(3 + i): USD0 for i in range(SLOTS)})
    fmts.update({c_eq: USD0, c_cash: USD0, c_port: USD0, c_pidx: NUM2, c_bidx: NUM2,
                 c_pret: PCT, c_bret: PCT, c_ppk: USD0, c_pdd: PCT, c_bpk: USD,
                 c_bdd: PCT})
    for c, fmt in fmts.items():   # column-level formats keep the file small
        hist.column_dimensions[c].number_format = fmt
    hist.freeze_panes = "B4"
    for c in range(1, 14 + SLOTS):
        hist.column_dimensions[col(c)].width = 13

    # ----------------------------------------------------------- Dashboard
    title(dash, "Portfolio Dashboard")
    dash["A2"] = f'={"Settings!B4"}&"  |  Benchmark: "&{BENCH}&"  |  Since "&TEXT({START},"yyyy-mm-dd")'
    dash["A2"].font = Font(italic=True, color="666666")

    H = lambda c: f"History!{c}{H_FIRST}:{c}{H_LAST}"
    Hr = lambda c: f"History!{c}{nxt}:{c}{H_LAST}"
    kpis = [
        ("Performance", None, None),
        ("Portfolio value", f"=Holdings!H{port_row}", USD0),
        ("Starting capital", f"={CASH0}", USD0),
        ("Total return", f"=Holdings!H{port_row}/{CASH0}-1", PCT),
        ("Benchmark return", f"=IFERROR(GOOGLEFINANCE({BENCH})/Prices!B{H_FIRST}-1,\"\")", PCT),
        ("Excess return vs benchmark", "=B7-B8", PCT),
        ("Trading days tracked", f"=COUNT({H('A')})", "0"),
        ("Risk-adjusted (daily returns, annualized)", None, None),
        ("Beta vs benchmark", f"=IFERROR(SLOPE({Hr(c_pret)},{Hr(c_bret)}),\"\")", NUM2),
        ("Jensen's alpha (annual)",
         f"=IFERROR((AVERAGE({Hr(c_pret)})-{RF}/252-B12*(AVERAGE({Hr(c_bret)})-{RF}/252))*252,\"\")",
         PCT),
        ("Correlation with benchmark", f"=IFERROR(CORREL({Hr(c_pret)},{Hr(c_bret)}),\"\")", NUM2),
        ("Portfolio volatility", f"=IFERROR(STDEV({Hr(c_pret)})*SQRT(252),\"\")", PCT),
        ("Benchmark volatility", f"=IFERROR(STDEV({Hr(c_bret)})*SQRT(252),\"\")", PCT),
        ("Sharpe ratio",
         f"=IFERROR((AVERAGE({Hr(c_pret)})*252-{RF})/B15,\"\")", NUM2),
        ("Drawdown", None, None),
        ("Max drawdown (portfolio)", f"=MIN({H(c_pdd)})", PCT),
        ("Max drawdown (benchmark)", f"=MIN({H(c_bdd)})", PCT),
        ("Current drawdown", f"=IFERROR(INDEX({H(c_pdd)},B10),\"\")", PCT),
    ]
    for i, (label, formula, fmt) in enumerate(kpis, start=4):
        a = dash.cell(row=i, column=1, value=label)
        if formula is None:
            a.font = Font(bold=True, color="FFFFFF")
            a.fill = HEADER_FILL
            dash.cell(row=i, column=2).fill = HEADER_FILL
            continue
        b = dash.cell(row=i, column=2, value=formula)
        b.number_format = fmt
        b.font = BOLD
        for c in (a, b):
            c.fill = KPI_FILL
            c.border = BOX

    # Sector allocation, including cash
    srow = 4
    header(dash, srow, ["Sector allocation", "Weight", "Value"], col=4)
    s0, s1 = srow + 1, srow + SLOTS
    dash.cell(row=s0, column=4,
              value=f'=SORT(UNIQUE(FILTER(Holdings!C{HOLD_FIRST}:C{HOLD_LAST},'
                    f'Holdings!A{HOLD_FIRST}:A{HOLD_LAST}<>"")))')
    dash.cell(row=s0, column=6,
              value=f'=ARRAYFORMULA(IF(D{s0}:D{s1}="",,SUMIF(Holdings!C{HOLD_FIRST}:C{HOLD_LAST},'
                    f'D{s0}:D{s1},Holdings!H{HOLD_FIRST}:H{HOLD_LAST})))')
    dash.cell(row=s0, column=5,
              value=f'=ARRAYFORMULA(IF(D{s0}:D{s1}="",,F{s0}:F{s1}/Holdings!H{port_row}))')
    cash_s = s1 + 1
    dash.cell(row=cash_s, column=4, value="Cash")
    dash.cell(row=cash_s, column=5, value=f"=Holdings!I{cash_row}")
    dash.cell(row=cash_s, column=6, value=f"=Holdings!H{cash_row}")
    for r in range(s0, cash_s + 1):
        dash.cell(row=r, column=5).number_format = PCT
        dash.cell(row=r, column=6).number_format = USD0
        for c in (4, 5, 6):
            dash.cell(row=r, column=c).border = BOX

    widths(dash, {"A": 40, "B": 16, "C": 3, "D": 26, "E": 10, "F": 14})
    dash.freeze_panes = "A4"

    # Charts
    n = 260  # rows plotted; blanks beyond the last trading day are skipped
    perf = LineChart()
    perf.title = "Growth of 100: portfolio vs benchmark"
    perf.height, perf.width = 8, 18
    perf.y_axis.title = "Index (start = 100)"
    perf.add_data(Reference(hist, min_col=3 + SLOTS + 3, max_col=3 + SLOTS + 4,
                            min_row=3, max_row=3 + n), titles_from_data=True)
    perf.set_categories(Reference(hist, min_col=1, min_row=4, max_row=3 + n))
    dash.add_chart(perf, "H4")

    dd = AreaChart()
    dd.title = "Portfolio drawdown"
    dd.height, dd.width = 6, 18
    dd.add_data(Reference(hist, min_col=3 + SLOTS + 8, min_row=3, max_row=3 + n),
                titles_from_data=True)
    dd.set_categories(Reference(hist, min_col=1, min_row=4, max_row=3 + n))
    dd.legend = None
    dash.add_chart(dd, "H21")

    pie = PieChart()
    pie.title = "Sector allocation"
    pie.height, pie.width = 8, 12
    pie.add_data(Reference(dash, min_col=5, min_row=srow, max_row=cash_s), titles_from_data=True)
    pie.set_categories(Reference(dash, min_col=4, min_row=s0, max_row=cash_s))
    dash.add_chart(pie, "A24")

    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
