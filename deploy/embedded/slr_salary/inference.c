#include "model_coeffs.h"

// Compute salary from years of experience using simple linear regression
// Salary = intercept + coef * years_experience
double slr_predict_salary(double years_experience)
{
    return SLR_INTERCEPT +
           (SLR_COEF_YEARS_EXPERIENCE * years_experience);
}
