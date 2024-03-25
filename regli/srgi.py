import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Iterable
from scipy.interpolate import PchipInterpolator, make_interp_spline


def flat_to_mesh(x_flat: npt.NDArray) -> list[npt.NDArray]:
    # determine ndim
    ndim = x_flat.shape[1]
    # determine grid for each dimension
    grid = [np.unique(x_flat[:, idim]) for idim in range(ndim)]
    # construct mesh (a list of cubes)
    mesh = np.meshgrid(*grid, indexing="ij")
    return mesh


def mesh_to_flat(*xmi: npt.NDArray) -> npt.NDArray:
    # get shape
    shape = xmi[0].shape
    # determine nelement and ndim
    nelement = np.prod(shape)
    ndim = len(xmi)  # ==len(shape)
    # construct flat
    flat = np.zeros((nelement, ndim), dtype=float)
    # fill each dim
    for idim in range(ndim):
        flat[:, idim] = xmi[idim].flatten()
    return flat


class SemiRegularGridInterpolator:
    def __init__(
        self,
        param: npt.NDArray,
        data: npt.NDArray,
        method: str = "pchip",
        round_decimals: int = 4,
        fix_method: str = "pchip",
    ):
        _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, "pchip": 3}
        _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
        _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

        if not hasattr(param, "grid_ndim"):
            param = np.asarray(param)
        if not hasattr(data, "grid_ndim"):
            data = np.asarray(data)

        # round parameters if necessary
        if round_decimals is None:
            # do not round param
            self.param = param
        elif type(round_decimals) == int:
            self.param = np.round(param, round_decimals)
        elif isinstance(round_decimals, Iterable):
            self.param = np.empty_like(param, dtype=float)
            for idim in range(self.grid_ndim):
                self.param[:, idim] = np.round(param[:, idim], decimals=round_decimals)
        else:
            raise ValueError("Invalid value for ``round_decimals``!")

        # determine ndim / dims / shape
        self.npnt = param.shape[0]

        # grid / feature / full
        self.grid_ndim = param.shape[1]
        self.grid = [np.unique(self.param[:, idim]) for idim in range(self.grid_ndim)]
        self.feature_ndim = data.ndim - 1
        self.full_ndim = self.grid_ndim + self.feature_ndim

        # grid
        self.grid_dims = tuple(range(0, self.grid_ndim))
        self.grid_shape = tuple(len(self.grid[idim]) for idim in range(self.grid_ndim))
        # feature
        self.feature_dims = tuple(range(self.grid_ndim, self.full_ndim))
        self.feature_shape = data.shape[1:]
        # full
        self.full_dims = tuple(range(self.full_ndim))
        self.full_shape = (*self.grid_shape, *self.feature_shape)

        # construct grid data
        self.data = data
        self.grid_data = np.full(self.full_shape, fill_value=np.nan, dtype=float)
        # initialize grid_data
        print("Initialize self.grid_data")
        for ipnt in range(self.npnt):
            indices = self.find_grid_indices(self.grid, self.param[ipnt])
            print(f" - Initialize data for indices = {indices}")
            self.grid_data[*indices, ...] = self.data[ipnt]
        # generate grid mask: True for infinite values

        # to be fixed
        # self.to_be_fixed =
        self.fix_method = fix_method
        self.method = method

    @property
    def grid_data_mask(self):
        """Mask for grid data."""
        return ~np.isfinite(self.grid_data)

    @property
    def grid_mask(self):
        """Mask for grid."""
        return np.any(self.grid_data_mask, axis=self.feature_dims)

    @property
    def grid_mask_sum(self):
        """Sum of mask for grid."""
        return np.sum(self.grid_data_mask)

    @staticmethod
    def _do_spline_fit(x, y, pt, k, extrapolate=False):
        local_interp = make_interp_spline(x, y, k=k, axis=0)
        values = local_interp(pt)
        return values

    @staticmethod
    def _do_pchip(x, y, pt, k=None, extrapolate=False):
        local_interp = PchipInterpolator(x, y, axis=0, extrapolate=extrapolate)
        values = local_interp(pt)
        return values

    @staticmethod
    def find_grid_indices(grid, p):
        indices = []
        for _grid, _p in zip(grid, p):
            indices.append(np.where(_grid == _p)[0][0])
        return indices

    def __repr__(self):
        s = f"""<SemiRegularGridInterpolator>
- npnt:             {self.npnt}

- grid_ndim:        {self.grid_ndim}
- grid_dims:        {self.grid_dims}
- grid_shape:       {self.grid_shape}

- feature_ndim:     {self.feature_ndim}
- feature_dims:     {self.feature_dims}
- feature_shape:    {self.feature_shape}

- full_ndim:        {self.full_ndim}
- full_dims:        {self.full_dims}
- full_shape:       {self.full_shape}

- method:           {self.method}
"""
        return s

    def fix(
        self,
        fix_method="pchip",
        replace_all=False,
        interp_grid_dim=-1,
        extrapolate=False,
    ):
        """Fix each grid point."""
        # freeze grid mask for initial data
        grid_mask = self.grid_mask

        if fix_method == "pchip":
            _eval_func = self._do_pchip
        else:
            _eval_func = self._do_spline_fit

        # loop over bad points
        for indices in np.array(np.where(self.grid_mask)).T:
            p = [srgi.grid[i][_] for i, _ in enumerate(indices)]
            print(f" - Fixing data for indices = {indices} ...")
            # print(self.grid_data[*indices])

            # determine slice
            slc = list(indices)
            slc[interp_grid_dim] = slice(None)  # propose all
            index_valid_in_this_dimension = ~grid_mask[*slc]
            slc[interp_grid_dim] = index_valid_in_this_dimension  # clip nan
            # slc.append(Ellipsis)  # add feature dimensions

            # interpolate with finite data
            this_data_original = self.grid_data[*indices]
            this_data_interp = _eval_func(
                self.grid[interp_grid_dim][index_valid_in_this_dimension],
                self.grid_data[*slc, ...],  # ... for features
                self.grid[interp_grid_dim][indices[interp_grid_dim]],
                extrapolate=extrapolate,
            )

            # replace invalid part or whole
            if replace_all:
                self.grid_data[*indices] = this_data_interp
            else:
                self.grid_data[*indices] = np.where(
                    np.isfinite(this_data_original),
                    this_data_original,
                    this_data_interp,
                )

    def __call__(self, *args, **kwargs):
        pass


grid = [
    np.linspace(0, 1, 10),
    np.linspace(0, 1, 20),
    np.linspace(0, 1, 30),
]
param = np.hstack([_.reshape(-1, 1) for _ in np.meshgrid(*grid, indexing="ij")])
nddata = np.random.rand(10, 20, 30, 4, 5)
data = nddata.reshape(-1, 4, 5)
data[123, 2, 3] = np.nan
data[125, 0, 1] = np.nan
# print(param.shape, data.shape)
srgi = SemiRegularGridInterpolator(param, data)
print(srgi)

print(srgi.grid_mask_sum)
srgi.fix(interp_grid_dim=1)
print(srgi.grid_mask_sum)


#     #
#     print(slc)
#     srgi.grid_data[*slc].shape
#
#     # construct slice
#     slc = [*indices, Ellipsis]
#
#     srgi.grid_dims[interp_grid_dim]
#     # for _i in
#
#
# srgi.grid_mask_sum
# srgi.to_be_fixed


# PchipInterpolator()
# mask = np.zeros(100, dtype=bool)
# mask[10:20] = True
# x = np.linspace(0, 1, 100)
# y = np.random.rand(100)
# # x[mask] = -100
# y[mask] = -100
# xma = np.ma.MaskedArray(x, mask=mask, fill_value=-100)
# yma = np.ma.MaskedArray(y, mask=mask, fill_value=-100)
# y_interp = PchipInterpolator(xma, yma, extrapolate=False)(x)
#
# plt.plot(x, y, label="truth")
# plt.plot(x, y_interp, label="interp")
# plt.legend()


from scipy.spatial import ConvexHull

hull = ConvexHull(srgi.param)
point_inside_hull = hull.find_simplex(p) >= 0
