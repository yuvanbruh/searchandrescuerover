from setuptools import find_packages, setup

package_name = 'frontier_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuvan',
    maintainer_email='asrishith07@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
entry_points={
    'console_scripts': [
        'explorer = frontier_explorer.explorer:main',
        'utility = frontier_explorer.utility:main',
        'logs = frontier_explorer.logs:main',
        'mission2 = frontier_explorer.mission2:main',
    ],
},
)
