import math
import tkinter as tk
from typing import Callable, Dict, List, Optional


class BAIIPlusCalculator:
    """A Tkinter based clone of the BA II Plus financial calculator."""

    BUTTON_LAYOUT = [
        ["2ND", "N", "I/Y", "PV", "PMT", "FV"],
        ["CPT", "CE/C", "DEL", "INS", "\u2193", "ON/OFF"],
        ["CF", "NPV", "IRR", "AMORT", "P/Y", "C/Y"],
        ["STO", "RCL", "y^x", "1/x", "x^2", "\u221a"],
        ["7", "8", "9", "\u00f7", "LN", "e^x"],
        ["4", "5", "6", "\u00d7", "10^x", "LOG"],
        ["1", "2", "3", "-", "EE", "\u2211+"],
        ["0", ".", "+/-", "+", "ENTER", "="]
    ]

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BA II Plus (Python Clone)")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)

        self.display_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready")
        self.second_var = tk.StringVar(value=" ")

        self.accumulator: Optional[float] = None
        self.pending_operator: Optional[str] = None
        self.just_evaluated = False

        self.second_mode = False
        self.begin_mode = False
        self.pending_compute = False
        self.pending_store = False
        self.pending_recall = False

        self.periods_per_year = 12
        self.compounds_per_year = 12

        self.memory: Dict[int, float] = {i: 0.0 for i in range(10)}
        self.statistics_sum = 0.0
        self.statistics_count = 0

        self.tvm_values: Dict[str, Optional[float]] = {
            "N": None,
            "I/Y": None,
            "PV": None,
            "PMT": None,
            "FV": None,
        }

        self.cash_flows: List[float] = []
        self.cf_index = -1

        self._build_interface()

    def _build_interface(self) -> None:
        header = tk.Frame(self.root, bg="#131313", bd=2, relief="ridge")
        header.pack(fill="x", padx=12, pady=12)

        title = tk.Label(
            header,
            text="BA II Plus",
            font=("Helvetica", 20, "bold"),
            fg="#f0f0f0",
            bg="#131313",
        )
        title.pack(side=tk.LEFT, padx=(6, 0))

        subtitle = tk.Label(
            header,
            text="BUSINESS ANALYST",
            font=("Helvetica", 10),
            fg="#b5b5b5",
            bg="#131313",
        )
        subtitle.pack(side=tk.RIGHT, padx=6)

        display_frame = tk.Frame(self.root, bg="#0f0f0f", bd=4, relief="sunken")
        display_frame.pack(fill="x", padx=16, pady=(0, 16))

        self.display_label = tk.Label(
            display_frame,
            textvariable=self.display_var,
            anchor="e",
            font=("Consolas", 28),
            bg="#0f0f0f",
            fg="#88ff96",
            width=16,
        )
        self.display_label.pack(fill="x")

        indicator_frame = tk.Frame(self.root, bg="#1e1e1e")
        indicator_frame.pack(fill="x", padx=16, pady=(0, 8))

        second_indicator = tk.Label(
            indicator_frame,
            textvariable=self.second_var,
            font=("Helvetica", 10, "bold"),
            fg="#ffd34e",
            bg="#1e1e1e",
            anchor="w",
        )
        second_indicator.pack(side=tk.LEFT)

        status_label = tk.Label(
            indicator_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            fg="#d0d0d0",
            bg="#1e1e1e",
            anchor="e",
        )
        status_label.pack(side=tk.RIGHT)

        button_frame = tk.Frame(self.root, bg="#1e1e1e")
        button_frame.pack(padx=12, pady=(0, 12))

        for row_index, row in enumerate(self.BUTTON_LAYOUT):
            for col_index, label in enumerate(row):
                btn = tk.Button(
                    button_frame,
                    text=label,
                    width=6,
                    height=2,
                    font=("Helvetica", 11, "bold"),
                    fg="#f5f5f5" if label != "2ND" else "#1a1a1a",
                    bg=self._button_bg(label),
                    activebackground="#454545",
                    relief="raised",
                    command=lambda key=label: self.handle_key(key),
                )
                btn.grid(row=row_index, column=col_index, padx=3, pady=3, sticky="nsew")

        for col in range(len(self.BUTTON_LAYOUT[0])):
            button_frame.grid_columnconfigure(col, weight=1)

    def _button_bg(self, label: str) -> str:
        if label == "2ND":
            return "#ffd34e"
        if label in {"CPT", "CE/C", "DEL", "INS", "\u2193", "ON/OFF"}:
            return "#2f2f2f"
        if label in {"CF", "NPV", "IRR", "AMORT", "P/Y", "C/Y", "STO", "RCL"}:
            return "#343434"
        return "#3d3d3d"

    # --- Display helpers -------------------------------------------------
    def set_display(self, value: float) -> None:
        if math.isfinite(value):
            formatted = f"{value:,.10g}"
            self.display_var.set(formatted)
        else:
            self.display_var.set("Error")
        self.just_evaluated = True

    def get_display_value(self) -> float:
        text = self.display_var.get().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return 0.0

    def handle_key(self, label: str) -> None:
        if self.second_mode and label in self.SECONDARY_HANDLERS:
            handler = self.SECONDARY_HANDLERS[label]
            self.second_mode = False
            self.second_var.set(" ")
            handler(self)
            return

        handler = self.HANDLERS.get(label)
        if handler:
            handler(self)
        elif label.isdigit():
            self.append_digit(label)
        else:
            self.status_var.set(f"Unhandled key: {label}")

    # --- Command registrations ------------------------------------------
    def toggle_second(self) -> None:
        self.second_mode = not self.second_mode
        self.second_var.set("2ND" if self.second_mode else " ")
        self.status_var.set("2ND active" if self.second_mode else "Ready")

    def append_digit(self, digit: str) -> None:
        if self.just_evaluated or self.display_var.get() == "0":
            new_value = digit
        else:
            new_value = self.display_var.get() + digit
        self.display_var.set(new_value)
        self.just_evaluated = False

    def append_decimal(self) -> None:
        if self.just_evaluated:
            self.display_var.set("0.")
            self.just_evaluated = False
            return
        if "." not in self.display_var.get():
            self.display_var.set(self.display_var.get() + ".")

    def toggle_sign(self) -> None:
        value = self.display_var.get()
        if value.startswith("-"):
            self.display_var.set(value[1:])
        elif value != "0":
            self.display_var.set("-" + value)

    def clear_entry(self) -> None:
        self.display_var.set("0")
        self.just_evaluated = False
        self.status_var.set("Entry cleared")

    def clear_all(self) -> None:
        self.accumulator = None
        self.pending_operator = None
        self.pending_compute = False
        self.pending_store = False
        self.pending_recall = False
        self.second_mode = False
        self.display_var.set("0")
        self.just_evaluated = False
        self.second_var.set(" ")
        self.status_var.set("All clear")

    def delete_last(self) -> None:
        value = self.display_var.get()
        if self.just_evaluated:
            self.clear_entry()
            return
        if len(value) <= 1:
            self.display_var.set("0")
        else:
            self.display_var.set(value[:-1])

    def start_operation(self, operator: str) -> None:
        current = self.get_display_value()
        if self.pending_operator and not self.just_evaluated:
            current = self.evaluate_binary(self.pending_operator, self.accumulator or 0.0, current)
            self.set_display(current)
        self.accumulator = current
        self.pending_operator = operator
        self.just_evaluated = True
        self.status_var.set(f"Pending {operator}")

    def equals(self) -> None:
        if not self.pending_operator:
            return
        operand = self.get_display_value()
        result = self.evaluate_binary(self.pending_operator, self.accumulator or 0.0, operand)
        self.set_display(result)
        self.accumulator = None
        self.pending_operator = None
        self.status_var.set("Result computed")

    def evaluate_binary(self, operator: str, left: float, right: float) -> float:
        try:
            if operator == "+":
                return left + right
            if operator == "-":
                return left - right
            if operator in {"\u00d7", "*"}:
                return left * right
            if operator in {"\u00f7", "/"}:
                return left / right
            if operator == "y^x":
                return left ** right
        except ZeroDivisionError:
            self.status_var.set("Division by zero")
            return float("nan")
        return right

    def unary_operation(self, func, description: str) -> None:
        value = self.get_display_value()
        try:
            result = func(value)
            self.set_display(result)
            self.status_var.set(description)
        except ValueError:
            self.display_var.set("Error")
            self.status_var.set("Math domain error")
            self.just_evaluated = True

    def reciprocal(self) -> None:
        value = self.get_display_value()
        if value == 0:
            self.display_var.set("Error")
            self.status_var.set("Division by zero")
            self.just_evaluated = True
            return
        self.set_display(1 / value)
        self.status_var.set("Reciprocal")

    def toggle_begin_mode(self) -> None:
        self.begin_mode = not self.begin_mode
        mode = "BGN" if self.begin_mode else "END"
        self.status_var.set(f"Payments set to {mode}")

    def set_periods_per_year(self) -> None:
        value = int(self.get_display_value())
        if value <= 0:
            self.status_var.set("Invalid P/Y value")
            return
        self.periods_per_year = value
        self.status_var.set(f"P/Y = {value}")
        self.just_evaluated = True

    def set_compounds_per_year(self) -> None:
        value = int(self.get_display_value())
        if value <= 0:
            self.status_var.set("Invalid C/Y value")
            return
        self.compounds_per_year = value
        self.status_var.set(f"C/Y = {value}")
        self.just_evaluated = True

    def cpt_pressed(self) -> None:
        self.pending_compute = True
        self.status_var.set("Compute: choose a TVM key")

    def handle_tvm(self, key: str) -> None:
        if self.pending_compute:
            result = self.compute_tvm(key)
            if result is not None:
                self.set_display(result)
                self.status_var.set(f"Computed {key}")
            self.pending_compute = False
            return

        value = self.get_display_value()
        self.tvm_values[key] = value
        self.status_var.set(f"Stored {key} = {value}")
        if key == "N":
            self.just_evaluated = True

    def clear_tvm(self) -> None:
        for name in self.tvm_values:
            self.tvm_values[name] = None
        self.pending_compute = False
        self.display_var.set("0")
        self.just_evaluated = True
        self.status_var.set("TVM cleared")

    def compute_tvm(self, target: str) -> Optional[float]:
        values = self.tvm_values.copy()
        when = 1 if self.begin_mode else 0

        rate_percent = values.get("I/Y")
        if rate_percent is None:
            self.status_var.set("Set I/Y first")
            return None

        periodic_rate = (rate_percent / 100.0) / max(self.compounds_per_year, 1)
        n_value = values.get("N")
        if n_value is not None:
            n_periods = n_value
        else:
            n_periods = None

        pv = values.get("PV")
        pmt = values.get("PMT")
        fv = values.get("FV")

        try:
            if target == "FV":
                if None in (n_periods, pv, pmt):
                    raise ValueError
                return self._fv(periodic_rate, n_periods, pmt, pv, when)
            if target == "PV":
                if None in (n_periods, fv, pmt):
                    raise ValueError
                return self._pv(periodic_rate, n_periods, pmt, fv, when)
            if target == "PMT":
                if None in (n_periods, pv, fv):
                    raise ValueError
                return self._pmt(periodic_rate, n_periods, pv, fv, when)
            if target == "N":
                if None in (pv, fv, pmt):
                    raise ValueError
                return self._nper(periodic_rate, pmt, pv, fv, when)
            if target == "I/Y":
                if None in (n_periods, pv, fv, pmt):
                    raise ValueError
                rate = self._rate(n_periods, pmt, pv, fv, when)
                return rate * 100.0 * max(self.compounds_per_year, 1)
        except ValueError:
            self.status_var.set("Incomplete TVM values")
            return None

        self.status_var.set("Unsupported TVM target")
        return None

    @staticmethod
    def _fv(rate: float, nper: float, pmt: float, pv: float, when: int) -> float:
        if rate == 0:
            return -(pv + pmt * nper)
        factor = (1 + rate) ** nper
        return -(pv * factor + pmt * (1 + rate * when) / rate * (factor - 1))

    @staticmethod
    def _pv(rate: float, nper: float, pmt: float, fv: float, when: int) -> float:
        if rate == 0:
            return -(fv + pmt * nper)
        factor = (1 + rate) ** nper
        return -(fv + pmt * (1 + rate * when) / rate * (factor - 1)) / factor

    @staticmethod
    def _pmt(rate: float, nper: float, pv: float, fv: float, when: int) -> float:
        if rate == 0:
            return -(fv + pv) / nper
        factor = (1 + rate) ** nper
        return -rate * (fv + pv * factor) / ((1 + rate * when) * (factor - 1))

    @staticmethod
    def _nper(rate: float, pmt: float, pv: float, fv: float, when: int) -> float:
        if rate == 0:
            if pmt == 0:
                raise ValueError
            return -(fv + pv) / pmt
        base = pmt * (1 + rate * when) - fv * rate
        denom = pv * rate + pmt * (1 + rate * when)
        if base <= 0 or denom <= 0:
            raise ValueError
        return math.log(base / denom) / math.log(1 + rate)

    def _rate(self, nper: float, pmt: float, pv: float, fv: float, when: int) -> float:
        rate = 0.05
        for _ in range(100):
            f = pv * (1 + rate) ** nper + pmt * (1 + rate * when) / rate * ((1 + rate) ** nper - 1) + fv
            df = (
                pv * nper * (1 + rate) ** (nper - 1)
                + pmt * (1 + rate * when) / rate * nper * (1 + rate) ** (nper - 1)
                - pmt * (1 + rate * when) / (rate ** 2) * ((1 + rate) ** nper - 1)
                + pmt * when / rate * ((1 + rate) ** nper - 1)
            )
            if df == 0:
                break
            new_rate = rate - f / df
            if abs(new_rate - rate) < 1e-10:
                return new_rate
            rate = new_rate
        return rate

    def cf_store(self) -> None:
        value = self.get_display_value()
        self.cash_flows.append(value)
        self.cf_index = len(self.cash_flows) - 1
        self.status_var.set(f"CF[{self.cf_index}] = {value}")
        self.display_var.set(f"{value:,.10g}")
        self.just_evaluated = True

    def cf_next(self) -> None:
        if not self.cash_flows:
            self.status_var.set("No cash flows")
            return
        self.cf_index = min(len(self.cash_flows) - 1, self.cf_index + 1)
        value = self.cash_flows[self.cf_index]
        self.display_var.set(f"{value:,.10g}")
        self.status_var.set(f"Viewing CF[{self.cf_index}]")
        self.just_evaluated = True

    def cf_prev(self) -> None:
        if not self.cash_flows:
            self.status_var.set("No cash flows")
            return
        self.cf_index = max(0, self.cf_index - 1)
        value = self.cash_flows[self.cf_index]
        self.display_var.set(f"{value:,.10g}")
        self.status_var.set(f"Viewing CF[{self.cf_index}]")
        self.just_evaluated = True

    def cf_delete(self) -> None:
        if 0 <= self.cf_index < len(self.cash_flows):
            removed = self.cash_flows.pop(self.cf_index)
            self.status_var.set(f"Removed CF[{self.cf_index}] = {removed}")
            if self.cash_flows:
                self.cf_index = min(self.cf_index, len(self.cash_flows) - 1)
                self.display_var.set(f"{self.cash_flows[self.cf_index]:,.10g}")
            else:
                self.cf_index = -1
                self.display_var.set("0")
            self.just_evaluated = True
        else:
            self.status_var.set("No CF selected")

    def cf_insert(self) -> None:
        value = self.get_display_value()
        position = max(0, self.cf_index)
        self.cash_flows.insert(position, value)
        self.cf_index = position
        self.status_var.set(f"Inserted CF[{position}] = {value}")
        self.display_var.set(f"{value:,.10g}")
        self.just_evaluated = True

    def cf_clear(self) -> None:
        self.cash_flows.clear()
        self.cf_index = -1
        self.display_var.set("0")
        self.just_evaluated = True
        self.status_var.set("Cash flows cleared")

    def compute_npv(self) -> None:
        if not self.cash_flows:
            self.status_var.set("Enter cash flows first")
            return
        rate_percent = self.tvm_values.get("I/Y")
        if rate_percent is None:
            self.status_var.set("Set I/Y for NPV")
            return
        rate = rate_percent / 100.0
        npv = 0.0
        for i, value in enumerate(self.cash_flows):
            npv += value / ((1 + rate) ** i)
        self.set_display(npv)
        self.status_var.set("NPV computed")

    def compute_irr(self) -> None:
        if len(self.cash_flows) < 2:
            self.status_var.set("Need at least two cash flows")
            return
        rate = 0.1
        for _ in range(100):
            npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(self.cash_flows))
            derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(self.cash_flows))
            if derivative == 0:
                break
            new_rate = rate - npv / derivative
            if abs(new_rate - rate) < 1e-8:
                rate = new_rate
                break
            rate = new_rate
        self.set_display(rate * 100)
        self.status_var.set("IRR computed (% per period)")

    def amortize(self) -> None:
        if self.tvm_values.get("PMT") is None or self.tvm_values.get("N") is None:
            self.status_var.set("Set TVM values before AMORT")
            return
        period = int(self.get_display_value())
        n = int(self.tvm_values["N"] or 0)
        if period <= 0 or period > n:
            self.status_var.set("Invalid period")
            return
        rate_percent = self.tvm_values.get("I/Y") or 0.0
        rate = (rate_percent / 100.0) / max(self.compounds_per_year, 1)
        balance = self.tvm_values.get("PV") or 0.0
        pmt = self.tvm_values.get("PMT") or 0.0
        interest = 0.0
        principal = 0.0
        for i in range(1, period + 1):
            if self.begin_mode and i == 1:
                applied_payment = 0.0
            else:
                applied_payment = pmt
            interest_payment = balance * rate
            principal_payment = applied_payment - interest_payment
            balance += principal_payment
            if i == period:
                interest = interest_payment
                principal = principal_payment
        self.status_var.set(
            f"P{period}: INT={interest:.2f} PRN={principal:.2f} BAL={balance:.2f}"
        )
        self.display_var.set(f"{balance:,.10g}")
        self.just_evaluated = True

    def memory_store(self) -> None:
        self.pending_store = True
        self.status_var.set("STO: choose register 0-9")

    def memory_recall(self) -> None:
        self.pending_recall = True
        self.status_var.set("RCL: choose register 0-9")

    def handle_memory_digit(self, digit: str) -> None:
        index = int(digit)
        if self.pending_store:
            value = self.get_display_value()
            self.memory[index] = value
            self.pending_store = False
            self.status_var.set(f"Stored {value} in R{index}")
            self.just_evaluated = True
            return
        if self.pending_recall:
            value = self.memory.get(index, 0.0)
            self.display_var.set(f"{value:,.10g}")
            self.pending_recall = False
            self.status_var.set(f"Recalled R{index}")
            self.just_evaluated = True
            return
        self.append_digit(digit)

    def enter_pressed(self) -> None:
        value = self.get_display_value()
        self.status_var.set(f"Entered {value}")
        self.just_evaluated = True

    def ee_pressed(self) -> None:
        if self.just_evaluated:
            self.display_var.set("1e")
            self.just_evaluated = False
        else:
            self.display_var.set(self.display_var.get() + "e")

    def sigma_plus(self) -> None:
        value = self.get_display_value()
        self.statistics_sum += value
        self.statistics_count += 1
        self.status_var.set(
            f"\u03a3+: n={self.statistics_count} sum={self.statistics_sum:.4f}"
        )
        self.just_evaluated = True

    def power_toggle(self) -> None:
        self.root.destroy()

    HANDLERS: Dict[str, Callable[["BAIIPlusCalculator"], None]] = {}
    SECONDARY_HANDLERS: Dict[str, Callable[["BAIIPlusCalculator"], None]] = {}

    def run(self) -> None:
        self.root.mainloop()


def _register_handlers() -> None:
    calc = BAIIPlusCalculator
    calc.HANDLERS = {
        "2ND": calc.toggle_second,
        ".": calc.append_decimal,
        "+/-": calc.toggle_sign,
        "CE/C": calc.clear_entry,
        "DEL": calc.delete_last,
        "INS": calc.cf_insert,
        "\u2193": calc.cf_next,
        "ON/OFF": calc.power_toggle,
        "CPT": calc.cpt_pressed,
        "N": lambda self: self.handle_tvm("N"),
        "I/Y": lambda self: self.handle_tvm("I/Y"),
        "PV": lambda self: self.handle_tvm("PV"),
        "PMT": lambda self: self.handle_tvm("PMT"),
        "FV": lambda self: self.handle_tvm("FV"),
        "CF": calc.cf_store,
        "NPV": calc.compute_npv,
        "IRR": calc.compute_irr,
        "AMORT": calc.amortize,
        "P/Y": calc.set_periods_per_year,
        "C/Y": calc.set_compounds_per_year,
        "STO": calc.memory_store,
        "RCL": calc.memory_recall,
        "y^x": lambda self: self.start_operation("y^x"),
        "1/x": calc.reciprocal,
        "x^2": lambda self: self.unary_operation(lambda x: x ** 2, "Squared"),
        "\u221a": lambda self: self.unary_operation(math.sqrt, "Square root"),
        "LN": lambda self: self.unary_operation(math.log, "Natural log"),
        "e^x": lambda self: self.unary_operation(math.exp, "Exponential"),
        "10^x": lambda self: self.unary_operation(lambda x: 10 ** x, "10^x"),
        "LOG": lambda self: self.unary_operation(lambda x: math.log10(x), "Log base 10"),
        "EE": calc.ee_pressed,
        "\u2211+": calc.sigma_plus,
        "+": lambda self: self.start_operation("+"),
        "-": lambda self: self.start_operation("-"),
        "\u00d7": lambda self: self.start_operation("\u00d7"),
        "\u00f7": lambda self: self.start_operation("\u00f7"),
        "=": calc.equals,
        "ENTER": calc.enter_pressed,
    }

    # Digit handlers with memory behaviour
    for digit in "0123456789":
        calc.HANDLERS[digit] = calc.handle_memory_digit

    calc.SECONDARY_HANDLERS = {
        "PMT": calc.toggle_begin_mode,
        "FV": calc.clear_tvm,
        "CE/C": calc.clear_all,
        "CF": calc.cf_clear,
        "NPV": calc.cf_prev,
        "\u2193": calc.cf_prev,
        "C/Y": calc.cf_delete,
    }


def main() -> None:
    _register_handlers()
    app = BAIIPlusCalculator()
    app.run()


if __name__ == "__main__":
    main()
