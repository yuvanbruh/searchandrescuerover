from setuptools import find_packages, setup

package_name = 'semantic_perception'

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
                    'yolo_detector = semantic_perception.yolo_detector:main',
                    'yolo = semantic_perception.yolo:main',
                    'best = semantic_perception.best:main',
                    'semantic_adapter = semantic_perception.semantic_adapter:main',
                    

                    'semantic_database = semantic_perception.semantic_database:main',


        ],
    },
)
