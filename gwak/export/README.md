To build container run: 

```
apptainer build $GWAK_CONTAINER_ROOT/export.sif apptainer.def
```

To build local enviroment run `poetry install`

and use `poetry run python export/main.py` to export the trained model. 

The default model to export is `background`. 

Use `poetry run python export/main.py --project <desired-project>` 

to export the desired project. 
