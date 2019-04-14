from setuptools import setup

setup(name='raman_rabi',
      version='0.1',
      description='Parameter fitting for Raman-Rabi electron-nuclear spin flip-flop transitions in NV centers.',
      url='https://github.com/phys201/raman_rabi',
      author='Taylor Patti, Jamelle Watson-Daniels, Soumya Ghosh',
      author_email='soumya_ghosh@g.harvard.edu',
      license='GNU LGPL',
      packages=['raman_rabi'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
