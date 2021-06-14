from setuptools import setup, find_namespace_packages, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
      long_description = fh.read()

setup(name='adversarial',
      version='0.0.1',
      description='Anti Adversarial Networks',
      long_description=long_description,
      url='http://github.com/jorchard/anti_adv',
      author='Brian Cechmanek',
      author_email='bcechmanek@gmail.com',
      classifiers=[
            "Programming Language :: Python :: 3.9+",
            "Operating System :: OS Independent",
      ],
      python_requires=">=3.9",
      # packages=find_packages(where="adversarial"),
      # packages=find_namespace_packages(where="src"),
      packages=['adversarial',
                'adversarial.datasets',   
                'adversarial.metrics',    
            #     'adversarial.networks',
                'adversarial.plotting', 
      ],
      # package_dir={"": "src",
      #             }, 
      # package_dir={"": "adversarial",
      #             'adversarial.datasets': "adversarial/datasets",
      #             # 'advmetrics': "src/metrics",
      #             # 'advnetworks': "src/networks",
      #             # 'advplotting': "src/plotting"
      # },
      zip_safe=False,
)
