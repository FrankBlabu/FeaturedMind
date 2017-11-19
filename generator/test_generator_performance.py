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
import generator.tools

#----------------------------------------------------------------------------------------------------------------------
# Test generator run time
#
@generator.tools.timeit
def test_run_time (generator, runs):

    for i in range (runs):
        print('\rRun {run} / {total}'.format (run=(i + 1), total=runs), end='\r')
        image, mask = generator.generate ()

    print ()

#----------------------------------------------------------------------------------------------------------------------
# Profile generator execution
#
def test_profile (generator, runs):

    pr = cProfile.Profile ()
    pr.enable ()

    image, mask = generator.generate ()

    pr.disable ()

    stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
    stats.print_stats ()


#----------------------------------------------------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    generators = [ generator.background.NoisyRectBackgroundGenerator,
                   generator.background.ImageBackgroundGenerator,
                   generator.fixture.FixtureGenerator,
                   generator.sheetmetal.SheetMetalGenerator ]

    parser.add_argument ('-x', '--width',     type=int, default=512, help='Width of the generated image')
    parser.add_argument ('-y', '--height',    type=int, default=512, help='Height of the generated image')
    parser.add_argument ('-g', '--generator', action='store', choices=[generator.TYPE for generator in generators])
    parser.add_argument ('-p', '--profile',   action='store_true', default=False, help='Profile run')
    parser.add_argument ('-r', '--runs',      type=int, default=None, help='Number of runs for timing measurement')

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

    if args.profile:
        if args.runs is not None:
            test_profile (generator, args.runs)
        else:
            test_profile (generator, 1)

    elif args.runs is not None:
        test_run_time (generator, args.runs)
