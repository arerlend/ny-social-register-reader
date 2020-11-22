# NY Social Register Reader

Breaks down social register documents from NY.

Rough outline of script structure:
- Split lines of text
- Ignore splits that intercept the vertical dividers
- Do OCR on individual lines of text (Tesseract performs worse on large blocks)

Club affiliations are stored as constants.

Output will be written into pandas dataframes and will be savable as CSV.
