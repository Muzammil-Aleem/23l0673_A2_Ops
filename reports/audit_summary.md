# Dataset audit summary

Total files: 975 (ignored files outside class folders: 0)

## Counts per class

| class        |   files |   readable |   corrupt |   duplicates |
|:-------------|--------:|-----------:|----------:|-------------:|
| ai_generated |     539 |        539 |         0 |            1 |
| real         |     436 |        436 |         0 |            0 |


## Width / height / file size (bytes) per class

| class        |   ('width', 'min') |   ('width', 'median') |   ('width', 'max') |   ('height', 'min') |   ('height', 'median') |   ('height', 'max') |   ('file_size', 'min') |   ('file_size', 'median') |   ('file_size', 'max') |
|:-------------|-------------------:|----------------------:|-------------------:|--------------------:|-----------------------:|--------------------:|-----------------------:|--------------------------:|-----------------------:|
| ai_generated |                225 |                  1024 |               6000 |                 135 |                    800 |                4001 |                  11738 |                    199760 |            5.62618e+06 |
| real         |                183 |                   920 |               8495 |                 158 |                    720 |                7091 |                  11311 |                    144840 |            1.01954e+07 |


## Overall

|        |   width |   height |        file_size |
|:-------|--------:|---------:|-----------------:|
| min    |     183 |      135 |  11311           |
| median |    1024 |      768 | 169093           |
| max    |    8495 |     7091 |      1.01954e+07 |


## Colour modes

| class        |   L |   P |   RGB |   RGBA |
|:-------------|----:|----:|------:|-------:|
| ai_generated |   1 |  14 |   489 |     35 |
| real         |   6 |   9 |   407 |     14 |


## Formats

| class        |   JPEG |   PNG |
|:-------------|-------:|------:|
| ai_generated |    429 |   110 |
| real         |    403 |    33 |


## Problems

- corrupt / unreadable files: 0
- exact duplicate files (extra copies): 1
- files whose hash appears in BOTH classes: 0
