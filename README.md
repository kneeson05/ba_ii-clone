# BA-II Plus Python Clone

This project provides a desktop clone of the Texas Instruments BA II Plus financial calculator. The application is built with Python and Tkinter and recreates the original keypad layout together with core financial features such as time value of money calculations, cash-flow analysis, and basic scientific operations.

## Getting started

1. Ensure Python 3.11 (or later) is available.
2. Launch the calculator:
   ```bash
   python ba_ii_plus.py
   ```
3. The window replicates the BA II Plus layout. Click on the buttons to perform calculations. Use the **2ND** key to access alternate actions such as clearing TVM registers (2ND + FV) or toggling begin/end payment mode (2ND + PMT).

## Feature highlights

- **Time value of money**: Store any four of N, I/Y, PV, PMT, or FV and press **CPT** followed by the missing variable to compute it. Payments can be toggled between begin/end modes via **2ND + PMT**.
- **Cash flows**: Enter values with **CF**, navigate with the arrow key, compute NPV via **NPV** (using I/Y as the discount rate) and IRR with **IRR**.
- **Scientific operations**: Square, square root, reciprocal, natural log, exponential, power, and logarithmic functions.
- **Memory registers**: Store and recall with **STO** and **RCL**, using the digit keys to select registers 0-9.
- **Statistics**: Use **Σ+** to accumulate values into a running sum and count.

## Notes

- The calculator window closes when **ON/OFF** is pressed.
- Some advanced BA II Plus behaviours (e.g., full amortisation reports and editing within the display line) are simplified in this clone while preserving the look-and-feel of the original device.
