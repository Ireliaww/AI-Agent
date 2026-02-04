class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 0:
            # According to LeetCode constraints (0 <= x <= 2^31 - 1),
            # x will always be non-negative.
            # If negative inputs were allowed, one might raise an error
            # or define specific behavior (e.g., return 0 or -1).
            pass 

        if x == 0:
            return 0

        # Binary search range for the square root of x.
        # The square root of x will be between 1 and x (inclusive) for x >= 1.
        # For x=1, sqrt is 1. For x=4, sqrt is 2. For x=8, sqrt is 2.
        left, right = 1, x
        
        # 'ans' will store the largest integer 'mid' such that mid*mid <= x.
        ans = 0

        while left <= right:
            mid = left + (right - left) // 2
            
            # Calculate the square of mid.
            # In Python, integers handle arbitrary size, so mid*mid won't overflow.
            # In languages like C++/Java, one might use `long long` for `square`
            # or check `mid > x / mid` to prevent overflow.
            square = mid * mid

            if square <= x:
                # mid could be the answer, or we might find a larger one.
                # Store mid as a potential answer and search in the right half.
                ans = mid
                left = mid + 1
            else:
                # mid*mid is greater than x, so mid is too large.
                # Search in the left half.
                right = mid - 1
        
        return ans

if __name__ == "__main__":
    sol = Solution()

    # Basic test cases
    assert sol.mySqrt(0) == 0, "Test Case 1 Failed: x = 0"
    assert sol.mySqrt(1) == 1, "Test Case 2 Failed: x = 1"
    assert sol.mySqrt(4) == 2, "Test Case 3 Failed: x = 4"
    assert sol.mySqrt(8) == 2, "Test Case 4 Failed: x = 8 (floor(sqrt(8)) = 2)"
    assert sol.mySqrt(9) == 3, "Test Case 5 Failed: x = 9"
    assert sol.mySqrt(15) == 3, "Test Case 6 Failed: x = 15 (floor(sqrt(15)) = 3)"
    assert sol.mySqrt(16) == 4, "Test Case 7 Failed: x = 16"

    # Edge cases / Large numbers
    assert sol.mySqrt(2) == 1, "Test Case 8 Failed: x = 2"
    assert sol.mySqrt(3) == 1, "Test Case 9 Failed: x = 3"
    assert sol.mySqrt(2147395600) == 46340, "Test Case 10 Failed: x = 2147395600 (perfect square)"
    assert sol.mySqrt(2147483647) == 46340, "Test Case 11 Failed: x = 2147483647 (max int, non-perfect square)"
    assert sol.mySqrt(2147488281) == 46341, "Test Case 12 Failed: x = 2147488281 (perfect square just above max int's sqrt)"

    print("All test cases passed!")