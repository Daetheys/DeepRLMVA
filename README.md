# DeepRLMVA

### Create new environment

The project is configured to use the `deeprl` conda environment which is described in `environment.yml`. To install it,
open a terminal in the project root directory and run the following:

```bash
$ conda env create -f environment.yml
```

Then setup a new interpreter in the interpreter settings of PyCharm from the newly created conda environment
named `deeprl`.

/!\ if you work on windows the installing of jaxlib might fail (hardly supported on windows). Thus run this command to
install the library correctly :

```bash
pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
```

If that does not work, there might be an incompatibility between the latest version of jax and the latest version of
jaxlib. You will then need to download the latest compatible jaxlib version
on https://whls.blob.core.windows.net/unstable/index.html and run the command within the deeprl environment:

```bash
pip install PATH-TO-WHL-FILE/whl_file_with_package_version.whl
```

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

### To run Tests

```bash
$ python -m pytest 
```