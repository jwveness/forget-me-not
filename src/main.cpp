#include <iostream>
#include <set>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>

// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/progress.hpp>

#include "common.hpp"
#include "ptw.hpp"
#include "statcounter.hpp"
#include "dataseq.hpp"
#include "mixture.hpp"
#include "fmn.hpp"


// global time index
size_t gbl_time_index = 0;


// experimental parameters and their defaults
struct ExperimentParameters {
    int unique_params = 7;
    int n = 5000;
    int repeats = 200;
    double switch_pr = 0.005;
};
ExperimentParameters params;


/* write the usage information to a stream. */
static void displayUsage(int argc, const char *argv[], std::ostream &o) {

    if (argc <= 0) return;

    o << "Usage: " << argv[0]
      << " [random seed] [number of tasks] [switch probability] [data sequence length] [repeats]"
      << std::endl;
}


/* initialize the random number generation, allowing the option of using a fixed seed. */
static void initRandomSeeds(int argc, const char *argv[]) {

    // global seed used to produce test data sequences
    if (argc > 1 && atoi(argv[1]) != 0) {
        unsigned int seed = atoi(argv[1]);
        std::cout << "Using fixed data sequence seed: " << seed << std::endl;
        setDataSeqSeed(seed);
    } else {
        unsigned int seed = static_cast<unsigned int>(std::time(NULL));
        std::cout << "Using time-based data sequence seed: " << seed << std::endl;
    }

    std::cout << std::endl;
}


/* process the command line arguments. */
static void processCommandLineOptions(int argc, const char *argv[]) {

    // check to see if the user needs help
    if (argc >= 2) {
        std::string s = argv[1];
        if (s == "help" || s == "-help" || s == "--help") {
            displayUsage(argc, argv, std::cout);
            std::exit(0);
        }
    }

    // process the optional arguments
    for (int i=2; i < argc; i++) {

        int arg = std::atoi(argv[i]);
        if (arg == 0 && i != 3) {
            std::cerr << "error: invalid command line argument #" << i << std::endl;
            std::exit(1);
        }

        switch (i) {
            case 2: params.unique_params = arg; break;
            case 3: params.switch_pr = std::atof(argv[i]); break;
            case 4: params.n = arg;             break;
            case 5: params.repeats = arg;       break;
            default:
                std::cerr << "error: invalid command line argument" << std::endl;
                std::exit(1);
        }
    }

    // set the random seed for data generation
    initRandomSeeds(argc, argv);
}


/* create the compression methods */
static void createMethods(boost::ptr_vector<Compressor> &methods) {

    methods.clear();
    methods.push_back(new PTW<KTEstimator,15>());
    methods.push_back(new KTEstimator());
    methods.push_back(new PruningForgetMeNot<ReservoirWrapper<KTEstimator, 100>, 15>(15));
}


/* display the experiment header. */
static void displayHeader(std::ostream &o) {

    o << "----------------------------------------------------" << std::endl;
    o << "             The Mysterious Bag of Coins            " << std::endl;
    o << "----------------------------------------------------" << std::endl;
    o << "Experiment Parameters: "  << std::endl;
    o << "Number of coins:        " << params.unique_params << std::endl;
    o << "Switch probability:     " << params.switch_pr << std::endl;
    o << "Sequence length:        " << params.n << std::endl;
    o << "Repetitions:            " << params.repeats << std::endl;
    o << std::endl;
}


/* report summary statistics and 95% approximate CIs. */
static void reportSummaryStats(
    std::ostream &o,
    const std::vector<StatCounter> &stats,
    const StatCounter &oracle_stats,
    boost::ptr_vector<Compressor> &methods
) {

    o << "Estimated Mean Redundancy:" << std::endl;

    for (size_t i=0; i < methods.size(); ++i) {

        double ci95 = 1.96 * std::sqrt(stats[i].variance() / params.repeats);

        o << methods[i].fileExtension() << " = "
          << stats[i].mean() << " +- " << ci95
          << std::endl;
    }

    o << std::endl;
    double ci95 = 1.96 * std::sqrt(oracle_stats.variance() / params.repeats);
    o << "oracle loss = " << oracle_stats.mean() << " +-" << ci95 << std::endl;
    o << std::endl;
}


/* sanity check for probabilities. */
static void checkProbability(double p) {

    if (p < 0.0 || p > 1.0) {
        std::cout << "error: out of bounds probability." << std::endl;
        exit(1);
    }
}


/* sanity check for binary density. */
static void checkBinaryDensity(double p, double p_not) {

    // use a one sided test since it is okay if we sum to
    // less than 1.0 if we have a semi-measure
    double err = p + p_not - 1.0;

    if (err >= 0.001) {
        std::cout << "error: density " << err << " far from summing to 1." << std::endl;
        exit(1);
    }
}


/* application entry point. */
int main(int argc, const char *argv[]) {

    processCommandLineOptions(argc, argv);

    displayHeader(std::cout);

    // initialize methods and statistics tracking
    boost::ptr_vector<Compressor> all_methods;
    createMethods(all_methods);
    std::vector<StatCounter> stats(all_methods.size());
    StatCounter oracle_stats;

    // evaluate each method multiple times
    for (int i=0; i < params.repeats; ++i) {

        boost::ptr_vector<Compressor> methods;
        createMethods(methods);

        // generate a particular data sequence to evaluate *all* methods.
        // this implements a form of common random numbers variance reduction
        // in terms of estimating the difference in scores of each method.
        DataSequence ds(params.unique_params, params.n, params.switch_pr);

        // loop over each method
        for (size_t j=0; j < methods.size(); ++j) {

            // process the data sequence
            gbl_time_index = 0;
            double acc_loss = 0.0;

            for (size_t k=0; k < ds.size(); ++k) {
                gbl_time_index = k + 1;

                double p = methods[j].prob(ds[k]);
                double p_not = methods[j].prob(!ds[k]);

                checkProbability(p); checkProbability(p_not);
                checkBinaryDensity(p, p_not);

                acc_loss += -std::log(p);

                methods[j].update(ds[k]);
            }

            double redundancy = acc_loss - ds.entropy();
            stats[j].push(redundancy);
        }

        oracle_stats.push(ds.entropy());
    }

    reportSummaryStats(std::cout, stats, oracle_stats, all_methods);

    return 0;
}


