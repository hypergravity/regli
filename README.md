## regli
REgular Grid Linear Interpolator


## install
```
pip install git+git://github.com/hypergravity/regli
```

## test


```python
from regli.regli import test
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
from regli import Regli         # import Regli
x1 = np.linspace(-1, 1, 30)     # construct grid coordinates
x2 = np.linspace(-1, 1, 30)
x3 = np.linspace(-1, 1, 30)
regli = RegularGrid(x1, x2, x3) # initiate regli using coordinates
f = lambda _x1, _x2, _x3: _x1 + _x2 + _x3   # an arbitrary function of coordinates

flats = regli.flats             # regli.flats stores flattened coordinates of ND grid
values = np.array([f(*_) for _ in flats]).reshape(-1, 1)  # evaluate your function on flats
regli.set_values(values)        # set values for regli

regli(pos)                      # use any of the 3 ways to interpolate
regli.interpn(pos)              # method 1 is equivalent to 2
regli.interp3(pos)              # this is accelerated for 3D

```
