#include "model_coeffs_autogen.h"

// Predict Profit using multiple linear regression.
// Inputs must follow the same order as MLR_FEATURE_NAMES / MLR_COEFS.
double mlr_predict_profit(const double x[MLR_NUM_FEATURES])
{
    double y = MLR_INTERCEPT;
    for (int i = 0; i < MLR_NUM_FEATURES; i++)
    {
        y += MLR_COEFS[i] * x[i];
    }
    return y;
}
