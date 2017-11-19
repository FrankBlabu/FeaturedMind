#!/usr/bin/python3
#
# test_generator_performance.py - Test of the performance of the generator classes
#
# Frank Blankenburg, Nov. 2017
#

import argparse
import random
import cProfile
import pstats
import sys

import generator.background
import generator.fixture
import generator.sheetmetal


#----------------------------------------------------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    generators = [ generator.background.BackgroundGenerator,
                   generator.background.ImageBackgroundGenerator,
                   generator.fixture.FixtureGenerator,
                   generator.sheetmetal.SheetMetalGenerator ]

    parser.add_argument ('-x', '--width',     type=int, default=512, help='Width of the generated image')
    parser.add_argument ('-y', '--height',    type=int, default=512, help='Height of the generated image')
    parser.add_argument ('-g', '--generator', action='store', choices=[generator.TYPE for generator in generators])

    generator.background.BackgroundGenerator.add_to_args_definition (parser)

    args = parser.parse_args ()

    #
    # Instantiate generator class from command line arguments
    #
    generator = None
    for candidate in generators:
        if args.generator == candidate.TYPE:
            generator = candidate (args)

    if generator is None:
        raise RuntimeError ('Unknown generator type \'{typename}\''.format (typename=args.generator))

    pr = cProfile.Profile ()
    pr.enable ()

    #
    # Generate probe
    #
    image, mask = generator.generate ()

    pr.disable ()

    stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
    stats.print_stats ()
