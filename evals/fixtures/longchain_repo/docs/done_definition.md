# Definition of done

A rollup task is finished only when **all** of the following hold:

1. `src/rollup.py` defines a module-level constant `ROLLUP_VERSION` whose value
   is the string `"1.0"`.
2. `report.md` contains a Markdown table with exactly the header
   `| stage | total |` and one row per distinct stage.
3. Every per-stage total in `report.md` is a plain integer with no thousands
   separator and no decimal point.

A report that is merely present is not done; the header row is part of the
definition because downstream tooling parses it.
