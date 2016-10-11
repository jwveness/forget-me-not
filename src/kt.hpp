#ifndef __KT_HPP__
#define __KT_HPP__

#include "common.hpp"

#include <deque>
#include <sstream>


/* KT Estimator for binary memoryless sources */
class KTEstimator : public Compressor {

    public:

        KTEstimator() :
            m_log_kt(0.0)
        {
            m_counts[0] = 0;
            m_counts[1] = 0;
        }

        // the probability of seeing a particular symbol next
        virtual double prob(bit_t b) {

            static const double KT_Alpha = 0.5;
            static const double KT_Alpha2 = KT_Alpha + KT_Alpha;

            double num = double(m_counts[b]) + KT_Alpha;
            double den = double(m_counts[0] + m_counts[1]) + KT_Alpha2;
            return num / den;
        }

        // the logarithm of the probability of all processed bits
        virtual double logBlockProbability() const { return m_log_kt; }

        // process a new bit
        virtual void update(bit_t b) {
            m_log_kt += std::log(prob(b));
            m_counts[b]++;
        }

        // file extension
        virtual const char *fileExtension() const { return "KT"; }

        // create a clone from the current KT estimator
        virtual Compressor *clone() const { return new KTEstimator(*this); }

    private:

        double m_log_kt;
        uint64_t m_counts[2];
};


/** A KT estimator which remembers the time indices it was trained on,
    useful for vizualization. */
class TrackingKT : public Compressor {

    public:

        // the probability of a given symbol coming next under the model
        virtual double prob(bit_t b) {
            return m_kt.prob(b);
        }

        // the logarithm of the probability of all processed experience
        virtual double logBlockProbability() const {
            return m_kt.logBlockProbability();
        }

        // process a new piece of sensory experience
        virtual void update(bit_t b) {

            if (m_indices.empty() || m_indices.back() != gbl_time_index)
                m_indices.push_back(gbl_time_index);

            m_kt.update(b);
        }

        // file extension
        virtual const char *fileExtension() const { return "Tracking_KT"; }

        // makes a copy of the model
        virtual Compressor *clone() const { return new TrackingKT(*this); }

        // get access to the time indices the model was trained on
        const indices_t &indices() const { return m_indices; }

        void displayIndices(size_t seqlength, size_t window) const {

            // create the raw, undiscretized string representation
            std::ostringstream oss;
            size_t upto = 0;

            for (size_t t=1; t <= seqlength; t++) {
                if (upto < m_indices.size() && t == m_indices[upto]) {
                    oss << "*";
                    upto++;
                } else {
                    oss << "_";
                }
            }

            std::string raw = oss.str();

            // create discretized display string
            size_t stars = 0, dashes = 0;
            std::ostringstream oss2;

            for (auto it=raw.begin(); it != raw.end(); ++it) {

                if (*it == '*') stars++; else if (*it == '_') dashes++;

                if (stars + dashes == window) {
                    if (stars >= dashes)
                        oss2 << "*";
                    else
                        oss2 << "_";

                    stars = 0; dashes = 0;
                }
            }

            std::cout << oss2.str();
        }

    private:
        KTEstimator m_kt;
        std::vector<size_t> m_indices;
};

#endif // __KT_HPP__
