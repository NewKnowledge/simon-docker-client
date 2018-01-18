import os.path
import pickle
import typing
from http import client

import numpy  # type: ignore

from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils
from primitive_interfaces import base, transformer

__author__ = 'Distil'
__version__ = '1.0.0'


DOCKER_KEY = 'simon'

# It is useful to define these names, so that you can reuse it both
# for class type arguments and method signatures.
# This is just an example of how to define a more complicated input type,
# which is in fact more restrictive than what the primitive can really handle.
# One could probably just use "typing.Container" in this case, if accepting
# a wide range of input types.
Inputs = typing.Union[container.ndarray, container.List[float], container.List[container.List[float]]]
Outputs = container.List[float]


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """

    pass


class simon_docker(base.SingletonOutputMixin[Inputs, Outputs, None, Hyperparams], transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which sums all the values on input into one number.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '9c00d42d-382d-4177-a0e7-082da88a29c8',
        'version': __version__,
        'name': "Sum Values",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }, {
            'type': metadata_module.PrimitiveInstallationType.DOCKER,
            # A key under which information about a running container will be provided to the primitive.
            'key': DOCKER_KEY,
            'image_name': 'registry.gitlab.com/datadrivendiscovery/tests-data/summing',
            # Instead of a label, an exact hash of the image is required. This assures reproducibility.
            # You can see digests using "docker images --digests".
            'image_digest': 'sha256:07db5fef262c1172de5c1db5334944b2f58a679e4bb9ea6232234d71239deb64',
        }],
        # URIs at which one can obtain code for the primitive, if available.
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/sum.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.test.SumPrimitive',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
        # A metafeature about preconditions required for this primitive to operate well.
        'preconditions': [
            # Instead of strings you can also use available Python enumerations.
            metadata_module.PrimitivePrecondition.NO_MISSING_VALUES,
            metadata_module.PrimitivePrecondition.NO_CATEGORICAL_VALUES,
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        if DOCKER_KEY not in self.docker_containers:
            raise ValueError("Docker key '{docker_key}' missing among provided Docker containers.".format(docker_key=DOCKER_KEY))

    def _convert_value(self, value: typing.Any) -> typing.Union[numpy.ndarray, typing.List, typing.Any]:
        # Server does not know about container types, just standard numpy arrays and lists.
        if isinstance(value, container.ndarray):
            return value.view(numpy.ndarray)
        elif isinstance(value, container.List):
            return [self._convert_value(v) for v in value]
        else:
            return value

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # In the future, we should store here data in Arrow format into
        # Plasma store and just pass an ObjectId of data over HTTP.
        value = self._convert_value(inputs)
        data = pickle.dumps(value)

        # TODO: Retry if connection fails.
        #       This connection can sometimes fail because the service inside a Docker container
        #       is not yet ready, despite container itself already running. Primitive should retry
        #       a few times before aborting.

        # Primitive knows the port the container is listening on.
        connection = client.HTTPConnection(self.docker_containers[DOCKER_KEY], port=5001)
        # This simple primitive does not keep any state in the Docker container.
        # But if your primitive does have to associate requests with a primitive, consider
        # using Python's "id(self)" call to get an identifier of a primitive's instance.
        connection.request('POST', '/', data, {
            'Content-Type': 'multipart/form-data',
        })
        response = connection.getresponse()

        if response.status != 200:
            raise ValueError("Invalid HTTP response status: {status}".format(status=response.status))

        result = int(response.read())

        outputs = container.List[float]((result,), {
            'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List[float],
            'dimension': {
                'length': 1,
            },
        })
        outputs.metadata = outputs.metadata.update((metadata_module.ALL_ELEMENTS,), {
            'structural_type': float,
        })

        # Wrap it into default "CallResult" object: we are not doing any iterations.
        return base.CallResult(outputs)

    # Because numpy arrays do not contain shapes and dtype as part of their structural types,
    # we have to manually check those in metadata. In this case, just dtype which is stored as
    # "structural_type" on values themselves (and not the container or dimensions).
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]]) -> typing.Optional[metadata_module.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_module.DataMetadata, arguments['inputs'])
        dimension_index = 0
        while True:
            metadata = inputs_metadata.query((metadata_module.ALL_ELEMENTS,) * dimension_index)

            if 'dimension' not in metadata:
                break

            dimension_index += 1

        inputs_value_structural_type = metadata.get('structural_type', None)

        if inputs_value_structural_type is None:
            return None

        # Not a perfect way to check for a numeric type but will do for this example.
        # Otherwise check out "pandas.api.types.is_numeric_dtype".
        if not issubclass(inputs_value_structural_type, (float, int, numpy.number)):
            return None

        return output_metadata

