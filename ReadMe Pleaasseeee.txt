1. need dependencies ng python painstall gamit pip open cmd type:
   -> pip install PyMuPDF tqdm psutil

2. pleaaasseeee install ghostscript

3. also pleaassee add sa environment path
   - Under "User variables for %username%" Variable "Path"
   - Click "Edit"
   - Click "New"
   - Ilagay ung path na nasa baba:
      -> C:\Program Files\gs\gs10.05.0\bin\gswin64.exe
      -> C:\Program Files\gs\gs10.05.0\bin\gswin64c.exe

=====================================================

How to config:

- iedit ang "start_compression.bat" wag na galawin ang pypdfcompressor.py pleaaasseeeee
- bale ang syntax if want iopen sa cmd or gawa ka new batch file: 
   -> "python pypdfcompressor.py input output --quality 100 --fast"
      - WHERE:
         -> input (ung folder ng input)
         -> output (folder ng output)
         -> "--quality 100" ung quality so 0-100 bahala ka, quality 100 default
         -> "--fast" flag yan para no analyzing bs ung script tatagal tayo eh :)