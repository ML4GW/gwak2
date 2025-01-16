To build container run: 

```
apptainer build $GWAK_CONTAINER_ROOT/export.sif apptainer.def
```

Run `poetry install` in bash shell to build local enviroment. 

The default model to export is `background`. 

- Use `poetry run python export/main.py --project <desired-project>` 

to export the desired project. 

To access container simply run 
```
apptainer shell $GWAK_CONTAINER_ROOT/project.sif
```
