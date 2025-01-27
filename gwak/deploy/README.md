# Deploy your trained NN model #
The deploy application provides two major utilities:
- Exporting trained NN model to excutibles
- Producing inference results on sequential data

# Build enviroment #

```
cd gwak/gwak/deploy
poetry install
```
Use ```poetry add <package>``` to add new packages to the deploy application. 

# Use the following command to deploy your NN model #

Run ```poetry run python deploy/cli_export.py --config deploy/config/export.yaml``` to export model. 

Run ```poetry run python deploy/cli_infer.py --config deploy/config/infer.yaml``` to produce inference result. 
