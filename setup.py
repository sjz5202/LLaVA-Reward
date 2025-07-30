from setuptools import setup, find_packages

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

# flash-attn package depends on torch. Install torch before flash-attn
install_requires = [
    "torch==2.3.0", 
    "flash-attn==2.7.4.post1",  
]

install_requires += parse_requirements('requirements.txt')

setup(
    name="llava_reward",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.10',
)

