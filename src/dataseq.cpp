#include "dataseq.hpp"

#include <cassert>
#include <sstream>
#include <iomanip>

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>


// random number source
static boost::mt19937 randsrc(static_cast<unsigned int>(time(NULL)));
static boost::uniform_01<boost::mt19937 &> urng(randsrc);


/* sets the random seed for a data generating source. */
void setDataSeqSeed(unsigned int seed) {

    randsrc.seed(seed);
}


/** create a new data sequence given a fixed number of tasks with a
    new segment starting with probability p, of a given length. */
DataSequence::DataSequence(size_t unique_params, size_t length, double segment_switch_prob) :
    m_entropy(0.0)
{

    // generate possible set of biases
    std::vector<double> biases;
    for (size_t i = 0; i < unique_params; i++) {
        biases.push_back(urng());
    }
    boost::uniform_int<int> biasidx(0, static_cast<int>(biases.size()) - 1);

    // generate initial bias
    int bindx = biasidx(randsrc);
    double p = biases[bindx];

    do {

        // add the new coin flip
        bit_t b = urng() < p ? 1 : 0;
        m_data.push_back(b);
        m_entropy += b == 1 ? -std::log(p) : -std::log(1.0 - p);

        // check to see if we need to resample a new coin
        if (urng() < segment_switch_prob) {
            bindx = biasidx(randsrc);
            p = biases[bindx];
        }

    } while (m_data.size() < length);

    assert(length == m_data.size());
}


/* generate the segments and bias */
void DataSequence::genSegmentsAndBias(unsigned int segments, size_t length, size_t unique_params) {

    assert(length >= segments);
    assert(segments >= 1);

    // generate possible set of biases
    std::vector<double> biases;
    for (size_t i=0; i < unique_params; i++) {
        biases.push_back(urng());
    }

    boost::uniform_int<int> biasidx(0, static_cast<int>(biases.size())-1);

    // initial bias
    int bindx = biasidx(randsrc);
    m_bias.push_back(biases[bindx]);

    // generate the split points
    for (unsigned int i=0; i < segments - 1; i++) {

        // ensure split point doesn't already exist
        bool found = false;
        size_t sp;

        do {
            sp = static_cast<size_t>(urng() * double(length));
            found = std::find(
                m_locations.begin(),
                m_locations.end(),
                sp
            ) != m_locations.end();
        } while (found && sp != 0);

        m_locations.push_back(sp);

        // generate a unique bias
        int x = biasidx(randsrc);
        while (x == bindx)
            x = biasidx(randsrc);

        bindx = x;
        m_bias.push_back(biases[bindx]);
    }

    if (m_locations.size() > 0)
        std::sort(m_locations.begin(), m_locations.end());
}


/* create a new data sequence containing a particular number
   of piecewise stationary segments inside a sequence of a
   particular length  */
void DataSequence::genData(unsigned int segments, size_t length, size_t unique_params) {

    m_data.clear();
    m_bias.clear();
    m_locations.clear();
    m_entropy = 0.0;

    genSegmentsAndBias(segments, length, unique_params);

    // generate data and verbose description
    size_t c = 0;

    for (size_t i=0; i < length; i++) {

        if (c < m_locations.size() && i == m_locations[c]) {
            c++;
        }

        double bias = m_bias[c];
        bit_t b = urng() < bias ? 1 : 0;
        m_data.push_back(b);

        m_entropy += b == 1 ? -std::log(bias) : -std::log(1.0 - bias);
    }

    assert(length == m_data.size());
}


/** generate the verbose description of the data sequence. */
std::string DataSequence::info() const {

    std::ostringstream oss;

    oss << "Generating Source Description: " << std::endl;

    // describe the generation statistics
    size_t start = 0;
    for (size_t i=0; i < m_bias.size(); i++) {
        size_t loc = i == m_bias.size()-1 ? m_data.size() : m_locations[i];
        oss << std::fixed << std::setprecision(2);
        oss << "(" << start << "," << loc << "):" << m_bias[i] << " ";
        start = loc + 1;
    }

    oss << std::endl << std::endl;
    oss << "Empirical Source Description: " << std::endl;


    // describe the empirical statistics of the generated data
    start = 0;
    size_t ones = 0;
    size_t c = 0;

    for (size_t i=0; i < m_data.size(); i++) {

        size_t loc = c >= m_locations.size() ? m_data.size()+1 : m_locations[c];

        if (i == loc) {

            double emp_bias = double(ones) / double(i - start);
            oss << "(" << start << "," << i << "):" << emp_bias << " ";

            ones = 0;
            c++;
            start = i+1;
        }

        if (m_data[i] == 1) ones++;
    }

    double emp_bias = double(ones) / double(m_data.size() - start);
    oss << "(" << start << "," << m_data.size() << "):" << emp_bias << " ";

    return oss.str();
}


