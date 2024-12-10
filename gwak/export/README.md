To build container run: 

```
apptainer build $GWAK_CONTAINER_ROOT/export.sif apptainer.def
```

To build local enviroment run `poetry install`

and use `poetry run python export/main.py` to export the trained model.
