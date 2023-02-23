//
// Created by clementriu on 2/23/23.
//

#ifndef COLMAP_SRC_OPTIM_FAST_ACRANSAC_H_
#define COLMAP_SRC_OPTIM_FAST_ACRANSAC_H_

#include "acransac.h"
#include <iomanip>

namespace colmap {

// A histogram class.
// The Histogram object can keep a tally of values
// within a range, the range is arranged into some
// number of bins specified during construction.
// Any allocation of a Histogram object may throw
// a bad_alloc exception.
// Dedicated to the public domain.
// Jansson Consulting
// 2009-06-30, updated 2011-06-17 and 2011-08-03

// 2011-12-17 Modified by Pierre Moulon
//  - use vector array to avoid memory management
//  - add value by sequence with iterator

template<typename T>
class Histogram {
 public:
  // Construct a histogram that can count
  // within a range of values.
  // All bins of the histogram are set to zero.
  Histogram(
      const T &Start = T(0),
      const T &End = T(1),
      const size_t &nBins = 10) :
                                  Start(Start),
                                  End(End),
                                  nBins_by_interval(nBins / (End - Start)),
                                  nBins(nBins),
                                  overflow(0),
                                  underflow(0) {
    freq.resize(nBins, 0);
  }

  // Construct a histogram from a sequence of data
  template<typename DataInputIterator>
  void Add(DataInputIterator begin, DataInputIterator end) {
    for (DataInputIterator iter = begin; iter != end; ++iter)
      Add(static_cast<T>(*iter));
  }

  // Increase the count for the bin that holds a
  // value that is in range for this histogram or
  // the under-/overflow count if it is not in range.
  void Add(const T &x) {
    if (x > End) {
      overflow++;
      return;
    }
    if (x < Start)
      ++underflow;
    else {
      const size_t i(
          static_cast<size_t>(
              (x - Start) * nBins_by_interval));
      if (i < nBins) ++freq[i];
      else ++overflow;
    }
  }

  // Get the sum of all counts in the histogram.
  size_t GetTotalCount() const {
    return std::accumulate(freq.begin(), freq.end(), 0.0);
  }

  // Get the overflow count.
  size_t GetOverflow() const {
    return overflow;
  }

  // Get the underflow count.
  size_t GetUnderflow() const {
    return underflow;
  }

  // Get frequencies
  const std::vector<size_t> &GetHist() const { return freq; }

  // Get XbinsValue
  std::vector<T> GetXbinsValue() const {
    std::vector<T> vec_XbinValue(nBins, T(0));
    double val = (End - Start) / static_cast<double>(nBins - 1);
    for (size_t i = 0; i < nBins; ++i)
      vec_XbinValue[i] = (val * static_cast<double>(i) + Start);
    return vec_XbinValue;
  }

  // Get start
  double GetStart() const { return Start; }

  // Get End
  double GetEnd() const { return End; }

  // Text display of the histogram
  std::string ToString(const std::string &sTitle = "") const {
    std::ostringstream os;
    if (!sTitle.empty())
      os << "\n" << sTitle << "\n";
    const size_t n = freq.size();
    for (size_t i = 0; i < n; ++i) {
      os << std::setprecision(3)
         << static_cast<float>(End - Start) / n * static_cast<float>(i)
         << "\t|\t" << freq[i] << "\n";
    }
    if (!freq.empty())
      os << std::setprecision(3) << End << "\n";
    return os.str();
  }

 private:
  double Start, End, nBins_by_interval;
  size_t nBins; // number of bins
  std::vector<size_t> freq; // histogram
  size_t overflow, underflow; //count under/over flow
};


template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class FAST_ACRANSAC : public ACRANSAC<Estimator, SupportMeasurer, Sampler> {
 public:
  typedef typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report Report;
  typedef typename ACRANSAC<Estimator, SupportMeasurer, Sampler>::ErrorIndex ErrorIndex;

  explicit FAST_ACRANSAC(const ACRANSACOptions& options);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y,
                  size_t imagesDimensions[], const double scalingFactor);

 protected:
  std::pair<double, double> bestNFA(const std::vector<double> &e, double loge0,
                     double maxThreshold, const std::vector<float>& logc_n,
                     const std::vector<float>& logc_k) const;
};

template <typename Estimator, typename SupportMeasurer, typename Sampler>
FAST_ACRANSAC<Estimator, SupportMeasurer, Sampler>::FAST_ACRANSAC(
    const ACRANSACOptions& options)
    : ACRANSAC<Estimator, SupportMeasurer, Sampler>(options) {
  options.Check();

  // Determine max_num_trials based on assumed `min_inlier_ratio`.
  const size_t kNumSamples = 100000;
}

/// Find best NFA and number of inliers wrt square error threshold in e.
template <typename Estimator, typename SupportMeasurer, typename Sampler>
std::pair<double, double> FAST_ACRANSAC<Estimator, SupportMeasurer, Sampler>::bestNFA(const std::vector<double> &e,
                                            double loge0,
                                            double maxThreshold,
                                            const std::vector<float> &logc_n,
                                            const std::vector<float> &logc_k) const {
  const double multError = (Estimator::DistToPoint ? 1.0 : 0.5);

  const int nBins = 20;
  Histogram<double> histo(0.0f, maxThreshold, nBins);
  histo.Add(e.cbegin(), e.cend());

  std::pair<double, double> current_best_nfa(std::numeric_limits<double>::infinity(), 0.0);
  unsigned int cumulative_count = 0;
  const std::vector<size_t> &frequencies = histo.GetHist();
  const std::vector<double> residual_val = histo.GetXbinsValue();
  for (int bin = 0; bin < nBins; ++bin) {
    cumulative_count += frequencies[bin];
    if (cumulative_count > Estimator::kMinNumSamples
        && residual_val[bin] > std::numeric_limits<float>::epsilon()) {
      const double logalpha = ACRANSAC<Estimator, SupportMeasurer, Sampler>::logalpha0_[0]
                              + multError * log10(residual_val[bin]
                                                  + std::numeric_limits<float>::epsilon());
      const std::pair<double, double> current_nfa(loge0
                                                      + logalpha *
                                                            (double) (cumulative_count - Estimator::kMinNumSamples)
                                                      + logc_n[cumulative_count]
                                                      + logc_k[cumulative_count],
                                                  residual_val[bin]);
      // Keep the best NFA iff it is meaningful ( NFA < 0 ) and better than the existing one
      if (current_nfa.first < current_best_nfa.first)
        current_best_nfa = current_nfa;
    }
  }
  return current_best_nfa;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename FAST_ACRANSAC<Estimator, SupportMeasurer, Sampler>::Report
FAST_ACRANSAC<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y, size_t imagesDimensions[],
    const double scalingFactor) {
  Timer timerLOR;
  timerLOR.Start();
  CHECK_EQ(X.size(), Y.size());


  ACRANSAC<Estimator, SupportMeasurer, Sampler>::_alpha0Right = Estimator::pSigma(1, imagesDimensions, false);
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::logalpha0_[1] = log10(ACRANSAC<Estimator, SupportMeasurer, Sampler>::_alpha0Right);
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::_alpha0Left = Estimator::pSigma(1, imagesDimensions, true);
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::logalpha0_[0] = log10(ACRANSAC<Estimator, SupportMeasurer, Sampler>::_alpha0Left);

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  const double maxThreshold = (ACRANSAC<Estimator, SupportMeasurer, Sampler>::options_.sigmaMax > 0)
                                  ? ACRANSAC<Estimator, SupportMeasurer, Sampler>::options_.sigmaMax * ACRANSAC<Estimator, SupportMeasurer, Sampler>::options_.sigmaMax
                                  :  // Square max error
                                  std::numeric_limits<double>::infinity();

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  std::vector<double> indexedErrors(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  ACRANSAC<Estimator, SupportMeasurer, Sampler>::sampler.Initialize(num_samples);

  // Possible sampling indices (could change in the optimization phase)
  std::vector<int> vInliers;
  std::vector<typename Estimator::X_t> Xselected(num_samples);
  std::vector<typename Estimator::Y_t> Yselected(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    Xselected[i] = X[i];
    Yselected[i] = Y[i];
  }

  // Precompute log combi
  double loge0 = log10((double)Estimator::NbModels *
                       (num_samples - Estimator::kMinNumSamples));
  std::vector<float> vLogc_n, vLogc_k;
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::makelogcombi_n(num_samples, vLogc_n);
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::makelogcombi_k(Estimator::kMinNumSamples, num_samples, vLogc_k);

  size_t max_num_trials = ACRANSAC<Estimator, SupportMeasurer, Sampler>::options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, ACRANSAC<Estimator, SupportMeasurer, Sampler>::sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;
  size_t num_trials_reserve = max_num_trials / 10;
  dyn_max_num_trials -= num_trials_reserve;

  // Output parameters
  double minNFA = std::numeric_limits<double>::infinity();
  double errorMax = 0;
  int side = 0;

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    ACRANSAC<Estimator, SupportMeasurer, Sampler>::sampler.SampleXY(Xselected, Yselected, &X_rand, &Y_rand);
    bool better = false;
    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        ACRANSAC<Estimator, SupportMeasurer, Sampler>::estimator.Estimate(X_rand, Y_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      for (int i = 0; i < num_samples; ++i) {
        indexedErrors[i] = ACRANSAC<Estimator, SupportMeasurer, Sampler>::estimator.Residual(X[i], Y[i], sample_model) *
                               scalingFactor * scalingFactor;
      }
      CHECK_EQ(indexedErrors.size(), num_samples);

      // Most meaningful discrimination inliers/outliers
      std::pair<double, double> bestnfa =
          bestNFA(indexedErrors, loge0, maxThreshold, vLogc_n, vLogc_k);

      // Save as best subset if better than all previous subsets.
      if (bestnfa.first < minNFA) {  // A better model was found
        best_model = sample_model;
        better = true;
        minNFA = bestnfa.first;
        side = 0;
        vInliers.clear();
        for (int i = 0; i < num_samples; ++i) {
          if (indexedErrors[i] <= bestnfa.second) {
            vInliers.push_back(i);
          }
        }
        errorMax = bestnfa.second;  // Error threshold
      }
      // ORSA optimization: draw samples among best set of inliers so far
      if ((better && minNFA < 0) ||
          (report.num_trials + 1 == dyn_max_num_trials && num_trials_reserve)) {
        if (vInliers.empty()) {  // No model found at all so far
          dyn_max_num_trials++;  // Continue to look for any model, even not
                                 // meaningful
          num_trials_reserve--;
        } else {
          std::vector<int>::const_iterator itInlier = vInliers.begin();
          Xselected.resize(vInliers.size());
          Yselected.resize(vInliers.size());
          for (int i = 0; itInlier != vInliers.end(); itInlier++, i++) {
            Xselected[i] = X[*itInlier];
            Yselected[i] = Y[*itInlier];
          }
          if (num_trials_reserve) {
            dyn_max_num_trials = report.num_trials + 1 + num_trials_reserve;
            num_trials_reserve = 0;
          }
        }
      }
      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= ACRANSAC<Estimator, SupportMeasurer, Sampler>::options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  if (minNFA >= 0) vInliers.clear();

  report.model = best_model;

  // No valid model was found.
  if (minNFA >= 0) {
    return report;
  }

  report.success = true;
  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  std::vector<double> residuals(num_samples);
  ACRANSAC<Estimator, SupportMeasurer, Sampler>::estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), num_samples);

  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] * scalingFactor * scalingFactor <= errorMax) {
      report.inlier_mask[i] = true;
      report.support.num_inliers += 1;
      report.support.residual_sum += residuals[i];
    }
  }
  std::cout << "AC-RANSAC final threshold: " << sqrt(errorMax) << std::endl;
  report.ransacTimer = timerLOR.ElapsedSeconds();

  return report;
}


} // namespace colmap
#endif  // COLMAP_SRC_OPTIM_FAST_ACRANSAC_H_
