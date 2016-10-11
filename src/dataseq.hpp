#ifndef __DATASEQ_HPP__
#define __DATASEQ_HPP__

#include "common.hpp"

#include <vector>
#include <string>


// set the random data generating source to a particular seed
extern void setDataSeqSeed(unsigned int seed);


/* generate synthetic test data */
class DataSequence {

    public:

        // create a new data sequence given a fixed number of tasks with a
        // new segment starting with probability p, of a given length
        DataSequence(size_t unique_params, size_t length, double segment_switch_prob);

        // data accessor
        bit_t operator[](size_t idx) const { return m_data[idx]; }

        // length of data sequence
        size_t size() const { return m_data.size(); }

        // entropy of the data generating source in nats
        double entropy() const { return m_entropy; }

        // get a summary of the data sequence
        std::string info() const;

    private:

        // generate the piecewise stationary data
        void genData(unsigned int segments, size_t length, size_t unique_params);

        // generate the data segments and the bias for each segment
        void genSegmentsAndBias(unsigned int segments, size_t length, size_t unique_params);

        std::vector<double> m_bias;      // segment bias
        std::vector<size_t> m_locations; // split locations
        std::vector<bit_t> m_data;
        double m_entropy;
        std::string m_info;
};


#endif // __DATASEQ_HPP__
