## regli [![DOI](https://zenodo.org/badge/153591296.svg)](https://zenodo.org/badge/latestdoi/153591296)
REgular Grid Linear Interpolator, capable to deal with spectral library or similar model data.

## author
Bo Zhang, [bozhang@nao.cas.cn](mailto:bozhang@nao.cas.cn)

## home page
- [https://github.com/hypergravity/regli](https://github.com/hypergravity/regli)
- [https://pypi.org/project/regli/](https://pypi.org/project/regli/)

## install
- for the latest **stable** version: `pip install regli`
- for the latest **github** version: `pip install git+git://github.com/hypergravity/regli`
- for the Zenodo version: [https://zenodo.org/record/3461514](https://zenodo.org/record/3461514)

## test


```python
from regli import test
test()
```
output:
```
regli.interp3 x 10000: 0.5675415992736816 sec
regli.interpn x 10000: 2.5326197147369385 sec
rgi x 10000: 5.4028871059417725 sec
```

## doc
```python
# import Regli
from regli import Regli
import numpy as np

# construct grid coordinates
x1 = np.linspace(-1, 1, 30)     
x2 = np.linspace(-1, 1, 30)
x3 = np.linspace(-1, 1, 30)

# initiate regli using coordinates
regli = Regli(x1, x2, x3)

# an arbitrary function of coordinates (for demo)
f = lambda _x1, _x2, _x3: _x1 + _x2 + _x3

# regli.flats stores flattened coordinates of ND grid
flats = regli.flats
# evaluate your function on flats
values = np.array([f(*_) for _ in flats]).reshape(-1, 1)
# set values for regli
regli.set_values(values)        

regli(pos)                      # use any of the 3 ways to interpolate
regli.interpn(pos)              # method 1 is equivalent to 2
regli.interp3(pos)              # this is accelerated for 3D

```

## How to cite the code
BibTex:
```
@misc{https://doi.org/10.5281/zenodo.3461514,
  doi = {10.5281/zenodo.3461514},
  url = {https://zenodo.org/record/3461514},
  author = {Zhang, Bo},
  title = {hypergravity/regli: The Zenodo version},
  publisher = {Zenodo},
  year = {2019}
}
```
Please go to [https://search.datacite.org/works/10.5281/zenodo.3461514](https://search.datacite.org/works/10.5281/zenodo.3461514) for other formats.
