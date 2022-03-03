# DeepRLMVA


### Create new environment 

The project is configured to use the `deeprl` conda environment which is described in `environment.yml`.
To install it, open a terminal in the project root directory and run the following:
```bash
$ conda env create -f environment.yml
```
Then setup a new interpreter in the interpreter settings of PyCharm from the newly created conda environment 
named `deeprl`.


### Add new dependencies

#### Updating your conda environment when new dependencies are added to environment.yml

```bash
$ conda env update -f environment.yml  --prune
```

#### Updating the environment.yml file when you need new dependencies

```bash
$ conda env export --from-history > environment.yml
```

Then delete the last line in environment.yml with the prefix.