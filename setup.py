from distutils.core import setup

setup(name='SimonDockerClient',
    version='1.0.0',
    description='A wrapper around the simon docker image, launched locally',
    packages=['SimonDockerClient'],
    install_requires=["numpy","pandas","requests","typing"],
    entry_points = {
        'd3m.primitives': [
            'distil.simon_docker = SimonDockerClient:simon_docker'
        ],
    },
)
