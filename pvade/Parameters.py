import os
import yaml
import argparse

from mpi4py import MPI
from pandas import json_normalize
from jsonschema import validate


class SimParams:
    """Manages and validates all simulation parameters

    ``SimParams`` provides a set of methods to obtain user-input values from
    both input files and the command line. It also provides input validation
    via a yaml schema file. All parameters can later be access from PVade
    objects like ``params.parent_group.option`` for example,
    ``params.solver.dt``.


    Attributes:
        comm (MPI Communicator): An MPI communicator that will be shared by all PVade objects.
        flat_schema_dict (dict): A flattened dictionary version of the yaml schema file used to set default values and provide valid command-line options.
        input_dict (dict): The aggregated input dictionary containing all command-line updated values and input-file values.
        input_file_dict (dict): The dictionary representing the input file.
        num_procs (int): The total number of processors being used to solve the problem
        rank (int): A unique ID for each process in the range `[0, 1, ..., num_procs-1]`
        schema_dict (dict): The unflattened (nested) representation of the yaml schema dictionary.
    """

    def __init__(self, input_file_path=None):
        """Create a SimParams object

        This method begins by reading both the schema file and the
        user-specified file into dictionaries. It overrides entries in the
        user-specified dictionary with command-line inputs and finally
        validates the complete dictionary against the schema file.

        Args:
            input_file_path (str, optional): The complete path to the input yaml file defining the problem.
        """
        # Get MPI communicators
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # Open the schema file for reading and load its contents into a dictionary
        pvopt_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(pvopt_dir, "input_schema.yaml"), "r") as fp:
            self.schema_dict = yaml.safe_load(fp)

        # Flatten the schema dictionary to make navigation and CLI parsing easier
        self._flatten_schema_dict()

        # Initialize the single dict that will hold and combine ALL inputs
        self.input_dict = {}

        if input_file_path is not None:
            # Assert that the file exists
            assert os.path.isfile(
                input_file_path
            ), f"Could not find file '{input_file_path}', check it exists"

            # Open the input file for reading and load its contents into a dictionary
            with open(input_file_path, "r") as fp:
                if self.rank == 0:
                    print(f"Reading problem definition from {input_file_path}")
                self.input_file_dict = yaml.safe_load(fp)

            # Set values as specified by the input yaml file
            self._set_user_params_from_file()

            # Notes from WALID:
            # - Print a warning if the value set from the file isn't in the schema
            # - Print the values that don't fit
            # - everything you expect to store in parameters should be in the yaml schema

        else:
            # Assign all default parameters using the flattened schema

            # We can still support this, but maybe not the official getting
            # started strategy(provide a yaml file for demos)
            self._initialize_params_to_default()

        # Override any previous value with a command line input
        self._set_user_params_from_cli()

        # Check that the complete parameter spec conforms to the schema
        self._validate_inputs()

        # Store the nested dictionary as attributes on this object for easy
        # access e.g., params.domain.x_max instead of params['domain']
        # ['x_max']
        self._store_dict_as_attrs()

        self._add_derived_quantities()

    def _flatten_schema_dict(self):
        """Flatten the schema dictionary

        This method returns a flattened version of the schema dictionary where
        nested dictionary key:val pairs are separated with dots (`.`). For
        example::

            general:
                name: test_sim
                size:
                    x: 10
                    y: 20

        will be converted to::

            general.name: test_sim
            general.size.x: 10
            general.size.y: 20

        Args:
            None

        """
        flat_schema_raw = json_normalize(self.schema_dict, sep=".").to_dict()

        self.flat_schema_dict = {}

        for key in flat_schema_raw.keys():
            if "default" in key:
                root = key.split(".default")[0]

                short_key = root.split(".")
                short_key = [sk for sk in short_key if sk != "properties"]
                short_key = ".".join(short_key)

                self.flat_schema_dict[short_key] = {}

                for subkey in ["default", "type", "units", "description"]:
                    try:
                        val = flat_schema_raw[f"{root}.{subkey}"][0]
                    except:
                        val = None

                    self.flat_schema_dict[short_key][subkey] = val

    def _set_user_params_from_file(self):
        """Store the input-file supplied values

        This method stores the parameters supplied via input yaml file in the
        aggregated input dictionary.

        Args:
            None

        """
        flat_input_file_dict = json_normalize(self.input_file_dict, sep=".").to_dict()

        for key, val in flat_input_file_dict.items():
            path_to_input = key.split(".")
            self._set_nested_dict_value(
                self.input_dict, path_to_input, val[0], error_on_missing_key=False
            )

    def _initialize_params_to_default(self):
        """Initialize values to default

        This method uses the `default` entry associated with each parameter in
        the yaml schema to provide a sane value where none is supplied by the
        input yaml file.

        Args:
            None

        """
        for key, val in self.flat_schema_dict.items():
            path_to_input = key.split(".")
            self._set_nested_dict_value(
                self.input_dict,
                path_to_input,
                val["default"],
                error_on_missing_key=False,
            )

    def _set_user_params_from_cli(self):
        """Store values from command line

        This method uses the flattened schema dictionary, where nested options
        are stored as a one-layer dictionary with keys separated by dots
        (`.`), to generate a set of valid command line inputs that can be
        used to override options that would otherwise be set by the input
        yaml file or defaults. For example::

            python main.py --solver.dt 0.01

        will override whatever value of ``dt`` is present in the yaml file
        with the value 0.01. Note the double hyphen to set off an argument
        name, either with or without the assignment operator is allowed
        (e.g., both ``--solver.dt 0.01`` and ``--solver.dt=0.01`` are valid)

        Args:
            None

        """
        parser = argparse.ArgumentParser()

        ignore_list = ["input_file"]

        parser.add_argument(
            "--input_file",
            metavar="",
            type=str,
            help="The full path to the input file, e.g., 'intputs/my_params.yaml'",
        )

        for key, value in self.flat_schema_dict.items():

            help_message = f"{value['description']} (data type = {value['type']}, Units = {value['units']})"

            if value["type"] == "string":
                cli_type = str
            elif value["type"] == "number":
                cli_type = float
            elif value["type"] == "integer":
                cli_type = int
            else:
                cli_type = None

            parser.add_argument(
                f"--{key}", metavar="", type=cli_type, help=help_message
            )

        command_line_inputs, unknown = parser.parse_known_args()

        # Find any command line arguments that were used and replace those entries in params
        for key, value in vars(command_line_inputs).items():
            if key not in ignore_list and value is not None:
                path_to_input = key.split(".")
                self._set_nested_dict_value(
                    self.input_dict, path_to_input, value, error_on_missing_key=False
                )

                if self.rank == 0:
                    print(f"| Setting {key} = {value} from command line.")

        for key in unknown:
            if self.rank == 0:
                print(f"| Got unknown option {key}, skipping.")

    def _validate_inputs(self):
        """Validate the input dictionary

        This method uses the validate function from the jsonschema package to
        validate the complete user-supplied parameter set. This checks that
        variables are the expected type, fall within prescribed ranges, and
        also ensure that dependencies between variables are satisfied
        (e.g., if variable ``a`` is specified, then variable ``b`` must also
        be specified.)

        Args:
            None

        """
        # This compares the input dictionary against the yaml schema
        # to ensure all values are set correctly
        validate(self.input_dict, self.schema_dict)

        # self._pprint_dict(self.input_dict)

    def _store_dict_as_attrs(self):
        """Converts dictionary notation to nested object attributes

        This method re-structures the parameters dictionary into the slightly
        easier-to-navigate ``object.attribute`` format. For example, rather
        than having a parameters dictionary and accessing entries like
        ``params['general']['name']``, we create a parameters object with
        attributes accessed like ``params.general.name``.

        Args:
            None

        """
        flat_input_dict = json_normalize(self.input_dict, sep=".").to_dict()

        for key, val in flat_input_dict.items():
            path_to_input = key.split(".")
            self._rec_settattr(self, path_to_input, val[0], error_on_missing_key=False)

    def _set_nested_dict_value(
        self, parent_obj, path_to_input, value, error_on_missing_key=True
    ):
        """Set a nested value in a dictionary

        This method uses the flattened path to a dictionary value to create
        the nested structure/path and then store the value.

        Args:
            parent_obj (dict): The dictionary containing the value to be set
            path_to_input (list, str): A list of strings containing the key names to be used at each level
            value (int, float, object): The value to store at the final level of nesting
            error_on_missing_key (bool, optional): A flag to raise an error if any of the keys or sub-keys is not already present

        """
        key = path_to_input[-1]
        path_to_input = path_to_input[:-1]

        for p in path_to_input:
            if error_on_missing_key:
                assert (
                    p in parent_obj
                ), f"Could not find option '{p}' in set of valid inputs."
            parent_obj = parent_obj.setdefault(p, {})

        if error_on_missing_key:
            assert (
                key in parent_obj
            ), f"Could not find option '{key}' in set of valid inputs."
        parent_obj[key] = value

    def _rec_settattr(
        self, parent_obj, path_to_input, value, error_on_missing_key=True
    ):
        """Convert nested dictionary to object.attribute format

        This method recursively creates nested objects to hold attributes in the same manner as the nested dictionary.

        Args:
            parent_obj (object): The parent object that will contain this value (self when first called)
            path_to_input (list, str): A list of strings containing the attribute names to be used at each level
            value (int, float, object): The value to store at the final level of nesting
            error_on_missing_key (bool, optional): A flag to raise an error if any of the keys or sub-keys is not already present

        """

        class ParamGroup:

            """Empty class to enable nesting and attribute storage"""

            pass

        if len(path_to_input) == 1:
            if error_on_missing_key:
                assert hasattr(
                    parent_obj, path_to_input[0]
                ), f"Attribute '{path_to_input[0]}' is invalid."

            setattr(parent_obj, path_to_input[0], value)

        else:
            if error_on_missing_key:
                child_obj = getattr(parent_obj, path_to_input[0])
            else:
                child_obj = getattr(parent_obj, path_to_input[0], ParamGroup())

            setattr(parent_obj, path_to_input[0], child_obj)
            parent_obj = child_obj
            self._rec_settattr(
                parent_obj,
                path_to_input[1:],
                value,
                error_on_missing_key=error_on_missing_key,
            )

    def _pprint_dict(self, d, indent=0):
        """Pretty print a dictionary

        Print a dictionary where nesting is visualized with indentation.
        Useful for validation and debugging.

        Args:
            d (dict): The dictionary to print
            indent (int, optional): The amount of indentation to use for this level, default=0

        """
        for key, val in d.items():
            for k in range(indent):
                print("|   ", end="")

            if isinstance(val, dict):
                print(f"{key}:")
                indent += 1
                self._pprint_dict(val, indent)
                indent -= 1

            else:
                print(f"{key}: {val} ({type(val)})")

    def _add_derived_quantities(self):
        """Add derived quantities for convenience

        For convenience, it is helpful to store certain derived quantities in
        the SimParams object that are not explicitly provided by the user.
        For example, ``params.solver.t_steps`` is the integer number of steps
        implied by the user-provided entries for ``t_final`` and ``dt``. Such
        derived and/or convenience quantities are accumulated here. Note that
        no error checking is performed on these values since they do not
        appear in the yaml schema file, the expectation being that valid
        quantities up till this point will result in valid derived values.

        Args:
            None

        """
        self.solver.t_steps = int(self.solver.t_final / self.solver.dt)
        self.solver.save_xdmf_interval_n = int(
            self.solver.save_xdmf_interval / self.solver.dt
        )
        self.solver.save_text_interval_n = int(
            self.solver.save_text_interval / self.solver.dt
        )

        if hasattr(self.domain, "z_min"):
            self.domain.dim = 3
        else:
            self.domain.dim = 2
