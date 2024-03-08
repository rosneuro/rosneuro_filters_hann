#include <gtest/gtest.h>
#include "Hann.hpp"

namespace rosneuro {

class HannTestSuite : public ::testing::Test {
    public:
        HannTestSuite() {}
        ~HannTestSuite() {}
        void SetUp() { hann_filter = new Hann <double>(); }
        void TearDown() { delete hann_filter; }
        Hann <double>* hann_filter;
};

TEST_F(HannTestSuite, Constructor) {
    ASSERT_EQ(hann_filter->name_, "hann");
    ASSERT_FALSE(hann_filter->is_window_set_);
}

TEST_F(HannTestSuite, Configure) {
    ASSERT_TRUE(hann_filter->configure());
}

TEST_F(HannTestSuite, ApplyWithValidWindow) {
    DynamicMatrix<double> inputMatrix(3, 3);
    inputMatrix << 1, 2, 3,
                   4, 5, 6,
                   7, 8, 9;
    DynamicMatrix<double> result = hann_filter->apply(inputMatrix);

    ASSERT_EQ(result.rows(), inputMatrix.rows());
    ASSERT_EQ(result.cols(), inputMatrix.cols());

    DynamicMatrix<double> expected(3, 3);
    expected << 0, 0, 0,
                4, 5, 6,
                0, 0, 0;

    ASSERT_TRUE(result.isApprox(expected, 0.01));
}

TEST_F(HannTestSuite, ApplyWithoutWindow) {
    DynamicMatrix<double> inputMatrix(-1, 0);
    ASSERT_THROW(hann_filter->apply(inputMatrix), std::bad_alloc);
}

TEST_F(HannTestSuite, CreateWindowWithValidSamples) {
    ASSERT_TRUE(hann_filter->create_window(1));
}

TEST_F(HannTestSuite, CreateWindowWithInvalidSamples) {
    ASSERT_THROW(hann_filter->create_window(-1), std::bad_alloc);
}

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "test_hann");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}