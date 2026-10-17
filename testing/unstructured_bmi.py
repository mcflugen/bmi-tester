from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from bmipy import Bmi
from numpy.typing import NDArray


@dataclass(eq=False)
class Variable:
    values: NDArray
    grid: int
    location: str
    units: str
    intent: Literal["in", "inout", "out"]


@dataclass(eq=False)
class UnstructuredGrid:
    grid_type: str = "unstructured"
    x: NDArray | None = None
    y: NDArray | None = None
    z: NDArray | None = None
    edge_nodes: NDArray | None = None
    face_edges: NDArray | None = None
    face_nodes: NDArray | None = None
    nodes_per_face: NDArray | None = None


@dataclass
class Time:
    start: float
    end: float
    step: float
    current: float
    units: str


class UnstructuredBmi(Bmi):

    _variables: dict[str, Variable]
    _grids: dict[str, UnstructuredGrid]
    _time: Time

    def finalize(self) -> None:
        """Perform tear-down tasks for the model.

        Perform all tasks that take place after exiting the model's time
        loop. This typically includes deallocating memory, closing files and
        printing reports.
        """
        for var in self._variables.values():
            var.values.fill(0.0)

    def get_component_name(self) -> str:
        """Name of the component.

        Returns
        -------
        str
            The name of the component.
        """
        return "Unstructured BMI"

    def get_current_time(self) -> float:
        """Return the current time of the model.

        Returns
        -------
        float
            The current model time.
        """
        return self._time.current

    def get_end_time(self) -> float:
        """End time of the model.

        Returns
        -------
        float
            The maximum model time.
        """
        return self._time.end

    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid edges.
        """
        values = self._grids[grid].edge_nodes
        return 0 if values is None else len(values.reshape((-1, 2)))

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        """Get the edge-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        edge_nodes : ndarray of int, shape *(2 x nnodes,)*
            A numpy array to place the edge-node connectivity. For each edge,
            connectivity is given as node at edge tail, followed by node at
            edge head.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the edge-node connectivity.
        """
        values = self._grids[grid].edge_nodes
        if values is None:
            raise NotImplementedError("get_grid_edge_nodes")
        edge_nodes[:] = values
        return edge_nodes

    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid faces.
        """
        values = self._grids[grid].nodes_per_face
        return 0 if values is None else len(values)

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        """Get the face-edge connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_edges : ndarray of int
            A numpy array to place the face-edge connectivity.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-edge connectivity.
        """
        values = self._grids[grid].face_edges
        if values is None:
            raise NotImplementedError("get_grid_face_edges")
        face_edges[:] = values
        return face_edges

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        """Get the face-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_nodes : ndarray of int
            A numpy array to place the face-node connectivity. For each face,
            the nodes (listed in a counter-clockwise direction) that form the
            boundary of the face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-node connectivity.
        """
        values = self._grids[grid].face_nodes
        if values is None:
            raise NotImplementedError("get_grid_face_nodes")
        face_nodes[:] = values
        return face_nodes

    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid nodes.
        """
        return len(self._grids[grid].x)

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: np.ndarray
    ) -> np.ndarray:
        """Get the number of nodes for each face.

        Parameters
        ----------
        grid : int
            A grid identifier.
        nodes_per_face : ndarray of int, shape *(nfaces,)*
            A numpy array to place the number of nodes per face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the number of nodes per face.
        """
        values = self._grids[grid].nodes_per_face
        if values is None:
            raise NotImplementedError("get_grid_nodes_per_face")
        nodes_per_face[:] = values
        return nodes_per_face

    def get_grid_origin(self, grid: int, origin: np.ndarray) -> np.ndarray:
        """Get coordinates for the lower-left corner of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        origin : ndarray of float, shape *(ndim,)*
            A numpy array to hold the coordinates of the lower-left corner of
            the grid.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the coordinates of the grid's
            lower-left corner.
        """
        raise NotImplementedError("get_grid_origin")

    def get_grid_rank(self, grid: int) -> int:
        """Get number of dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Rank of the grid.
        """
        grid = self._grids[grid]
        return len([dim for dim in (grid.x, grid.y, grid.z) if dim is not None])

    def get_grid_shape(self, grid: int, shape: np.ndarray) -> np.ndarray:
        """Get dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        shape : ndarray of int, shape *(ndim,)*
            A numpy array into which to place the shape of the grid.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the grid's shape.
        """
        raise NotImplementedError("get_grid_shape")

    def get_grid_size(self, grid: int) -> int:
        """Get the total number of elements in the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Size of the grid.
        """
        return self.get_grid_node_count(grid)

    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> np.ndarray:
        """Get distance between nodes of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        spacing : ndarray of float, shape *(ndim,)*
            A numpy array to hold the spacing between grid rows and columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's spacing.
        """
        raise NotImplementedError("get_grid_spacing")

    def get_grid_type(self, grid: int) -> str:
        """Get the grid type as a string.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        str
            Type of grid as a string.
        """
        return self._grids[grid].grid_type

    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        """Get coordinates of grid nodes in the x direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        x : ndarray of float, shape *(nrows,)*
            A numpy array to hold the x-coordinates of the grid node columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's column x-coordinates.
        """
        values = self._grids[grid].x
        if values is None:
            raise NotImplementedError("get_grid_x")
        x[:] = values
        return x

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        """Get coordinates of grid nodes in the y direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        y : ndarray of float, shape *(ncols,)*
            A numpy array to hold the y-coordinates of the grid node rows.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's row y-coordinates.
        """
        values = self._grids[grid].y
        if values is None:
            raise NotImplementedError("get_grid_y")
        y[:] = values
        return y

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        """Get coordinates of grid nodes in the z direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        z : ndarray of float, shape *(nlayers,)*
            A numpy array to hold the z-coordinates of the grid nodes layers.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's layer z-coordinates.
        """
        values = self._grids[grid].z
        if values is None:
            raise NotImplementedError("get_grid_z")
        z[:] = values
        return z

    def get_input_item_count(self) -> int:
        """Count of a model's input variables.

        Returns
        -------
        int
          The number of input variables.
        """
        return len(self.get_input_var_names())

    def get_input_var_names(self) -> tuple[str]:
        """List of a model's input variables.

        Input variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The input variables for the model.

        Notes
        -----
        Standard Names enable the CSDMS framework to determine whether
        an input variable in one model is equivalent to, or compatible
        with, an output variable in another model. This allows the
        framework to automatically connect components.

        Standard Names do not have to be used within the model.
        """
        return tuple(
            sorted(
                name
                for name, var in self._variables.items()
                if var.intent.startswith("in")
            )
        )

    def get_output_item_count(self) -> int:
        """Count of a model's output variables.

        Returns
        -------
        int
          The number of output variables.
        """
        return len(self.get_output_var_names())

    def get_output_var_names(self) -> tuple[str]:
        """List of a model's output variables.

        Output variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The output variables for the model.
        """
        return tuple(
            sorted(
                name
                for name, var in self._variables.items()
                if var.intent.endswith("out")
            )
        )

    def get_start_time(self) -> float:
        """Start time of the model.

        Model times should be of type float.

        Returns
        -------
        float
            The model start time.
        """
        return self._time.start

    def get_time_step(self) -> float:
        """Return the current time step of the model.

        The model time step should be of type float.

        Returns
        -------
        float
            The time step used in model.
        """
        return self._time.step

    def get_time_units(self) -> str:
        """Time units of the model.

        Returns
        -------
        str
            The model time unit; e.g., `days` or `s`.

        Notes
        -----
        CSDMS uses the UDUNITS standard from Unidata.
        """
        return self._time.units

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """Get a copy of values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a *copy* of a model variable, with
        the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        ndarray
            The same numpy array that was passed as an input buffer.
        """
        dest[:] = self._variables[name].values
        return dest

    def get_value_at_indices(
        self, name: str, dest: np.ndarray, inds: np.ndarray
    ) -> np.ndarray:
        """Get values at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        inds : array_like
            The indices into the variable array.

        Returns
        -------
        array_like
            Value of the model variable at the given location.
        """
        dest[inds] = self._variables[name].values[inds]
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Get a reference to values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a reference to a model variable,
        with the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        array_like
            A reference to a model variable.
        """
        return self._variables[name].values

    def get_var_grid(self, name: str) -> int:
        """Get grid identifier for the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
          The grid identifier.
        """
        return self._variables[name].grid

    def get_var_itemsize(self, name: str) -> int:
        """Get memory use for each array element in bytes.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            Item size in bytes.
        """
        return self._variables[name].values.itemsize

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the a given variable is defined on.

        The grid topology can be composed of *nodes*, *edges*, and *faces*.

        *node*
            A point that has a coordinate pair or triplet: the most
            basic element of the topology.

        *edge*
            A line or curve bounded by two *nodes*.

        *face*
            A plane or surface enclosed by a set of edges. In a 2D
            horizontal application one may consider the word “polygon”,
            but in the hierarchy of elements the word “face” is most common.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The grid location on which the variable is defined. Must be one of
            `"node"`, `"edge"`, or `"face"`.

        Notes
        -----
        CSDMS uses the `ugrid conventions`_ to define unstructured grids.

        .. _ugrid conventions: http://ugrid-conventions.github.io/ugrid-conventions
        """
        return self._variables[name].location

    def get_var_nbytes(self, name: str) -> int:
        """Get size, in bytes, of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            The size of the variable, counted in bytes.
        """
        return self._variables[name].values.nbytes

    def get_var_type(self, name: str) -> str:
        """Get data type of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The Python variable type; e.g., ``str``, ``int``, ``float``.
        """
        return self._variables[name].values.dtype.name

    def get_var_units(self, name: str) -> str:
        """Get units of the given variable.

        Standard unit names, in lower case, should be used, such as
        ``meters`` or ``seconds``. Standard abbreviations, like ``m`` for
        meters, are also supported. For variables with compound units,
        each unit name is separated by a single space, with exponents
        other than 1 placed immediately after the name, as in ``m s-1``
        for velocity, ``W m-2`` for an energy flux, or ``km2`` for an
        area.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The variable units.

        Notes
        -----
        CSDMS uses the `UDUNITS`_ standard from Unidata.

        .. _UDUNITS: http://www.unidata.ucar.edu/software/udunits
        """
        return self._variables[name].units

    def initialize(self, config_file: str) -> None:
        """Perform startup tasks for the model.

        Perform all tasks that take place before entering the model's time
        loop, including opening files and initializing the model state. Model
        inputs are read from a text-based configuration file, specified by
        `config_file`.

        Parameters
        ----------
        config_file : str, optional
            The path to the model configuration file.

        Notes
        -----
        Models should be refactored, if necessary, to use a
        configuration file. CSDMS does not impose any constraint on
        how configuration files are formatted, although YAML is
        recommended. A template of a model's configuration file
        with placeholder values is used by the BMI.
        """
        with open(config_file):
            pass

    def set_value(self, name: str, src: np.ndarray) -> None:
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        src : array_like
            The new value for the specified variable.
        """
        self._variables[name].values[:] = src

    def set_value_at_indices(
        self, name: str, inds: np.ndarray, src: np.ndarray
    ) -> None:
        """Specify a new value for a model variable at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        inds : array_like
            The indices into the variable array.
        src : array_like
            The new value for the specified variable.
        """
        self._variables[name].values[inds] = src

    def update(self) -> None:
        """Advance model state by one time step.

        Perform all tasks that take place within one pass through the model's
        time loop. This typically includes incrementing all of the model's
        state variables. If the model's state variables don't change in time,
        then they can be computed by the :func:`initialize` method and this
        method can return with no action.
        """
        raise NotImplementedError("update")

    def update_until(self, time: float) -> None:
        """Advance model state until the given time.

        Parameters
        ----------
        time : float
            A model time later than the current model time.
        """
        raise NotImplementedError("update_until")


class UnstructuredNodesBmi(UnstructuredBmi):

    def __init__(self):
        self._variables = {
            "land_surface__elevation": Variable(
                values=np.ones(5, dtype=float),
                grid=0,
                location="node",
                units="m",
                intent="inout",
            )
        }

        self._grids = {
            0: UnstructuredGrid(
                x=np.array([0, 1, 2, 0, 1], float),
                y=np.array([0, 0, 0, 1, 1], float),
                z=np.array([2, 2, 2, 2, 2], float),
            ),
        }

        self._time = Time(
            start=0.0,
            end=10.0,
            step=3.0,
            current=0.0,
            units="s",
        )


class UnstructuredNodesEdgesBmi(UnstructuredNodesBmi):
    def __init__(self):
        self._variables = {
            "land_surface__elevation": Variable(
                values=np.ones(5, dtype=float),
                grid=0,
                location="node",
                units="m",
                intent="inout",
            ),
            "land_surface_air__temperature": Variable(
                values=np.ones(6, dtype=float),
                grid=0,
                location="edge",
                units="C",
                intent="in",
            ),
        }

        self._grids = {
            0: UnstructuredGrid(
                x=np.array([0, 1, 2, 0, 1], float),
                y=np.array([0, 0, 0, 1, 1], float),
                z=np.array([2, 2, 2, 2, 2], float),
                edge_nodes=np.asarray(
                    [
                        *[0, 1],
                        *[1, 2],
                        *[0, 3],
                        *[1, 4],
                        *[4, 2],
                        *[3, 4],
                    ],
                    dtype=int,
                ),
            ),
        }

        self._time = Time(
            start=0.0,
            end=10.0,
            step=3.0,
            current=0.0,
            units="s",
        )


class UnstructuredNodesEdgesFacesBmi(UnstructuredBmi):

    def __init__(self):
        self._variables = {
            "land_surface__elevation": Variable(
                values=np.ones(5, dtype=float),
                grid=0,
                location="node",
                units="m",
                intent="inout",
            ),
            "land_surface_air__temperature": Variable(
                values=np.ones(6, dtype=float),
                grid=0,
                location="edge",
                units="C",
                intent="in",
            ),
            "land_surface~10m-above_air_flow__speed": Variable(
                values=np.ones(2, dtype=float),
                grid=0,
                location="face",
                units="m/s",
                intent="out",
            ),
            "air__temperature": Variable(
                values=np.ones(1, dtype=float),
                grid=0,
                location="none",
                units="C",
                intent="inout",
            ),
        }

        self._grids = {
            0: UnstructuredGrid(
                x=np.array([0, 1, 2, 0, 1], float),
                y=np.array([0, 0, 0, 1, 1], float),
                z=np.array([2, 2, 2, 2, 2], float),
                edge_nodes=np.asarray(
                    [
                        *[0, 1],
                        *[1, 2],
                        *[0, 3],
                        *[1, 4],
                        *[4, 2],
                        *[3, 4],
                    ],
                    dtype=int,
                ),
                face_edges=np.asarray(
                    [
                        *[0, 3, 5, 2],
                        *[1, 4, 3],
                    ],
                    dtype=int,
                ),
                face_nodes=np.asarray(
                    [
                        *[0, 1, 4, 3],
                        *[1, 2, 4],
                    ],
                    dtype=int,
                ),
                nodes_per_face=np.asarray([4, 3], dtype=int),
            ),
        }

        self._time = Time(
            start=0.0,
            end=10.0,
            step=3.0,
            current=0.0,
            units="s",
        )
