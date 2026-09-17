# Dataset splits

The public split files (`train.csv`, `val.csv`, and `test.csv`) contain opaque `tile_id` values only.
They intentionally do **not** publish internal filesystem paths or spatial coordinate-bearing source filenames.

Authorized researchers should maintain a separate manifest outside the public repository with this format:

```csv
tile_id,path
tile_0001,/path/to/authorized/data/tile_0001.laz
```

Use `authorized_tile_manifest.example.csv` only as a format template. Do not commit the real authorized manifest if its paths or filenames disclose restricted archaeological locations.

The dataset loader accepts the manifest through `tile_manifest=` or the `MAYA_TILE_MANIFEST` environment variable.
