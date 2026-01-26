import heapq

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def solution(lists):
    """
    Merges k sorted linked lists into one sorted linked list using a min-heap (priority queue).

    Args:
        lists: A list of ListNode objects, where each ListNode is the head of a sorted linked list.

    Returns:
        The head of the merged sorted linked list.
    """
    min_heap = []
    dummy = ListNode(0)
    current = dummy
    
    # Counter for tie-breaking in heap comparison.
    # Python's heapq module needs elements to be comparable.
    # If two ListNode objects have the same 'val', Python might try to compare the ListNode objects themselves,
    # which are not inherently comparable unless custom __lt__ is defined.
    # Adding a unique counter as the second element in the tuple ensures distinctness for comparison,
    # preventing TypeError when values are identical.
    node_idx_counter = 0 

    # Push the head of each non-empty list into the min-heap.
    for head in lists:
        if head: # Only add non-empty lists' heads
            heapq.heappush(min_heap, (head.val, node_idx_counter, head))
            node_idx_counter += 1
            
    # While the heap is not empty, extract the smallest element.
    while min_heap:
        # Pop the node with the smallest value from the heap.
        val, idx, node = heapq.heappop(min_heap)
        
        # Append this node to the merged list.
        current.next = node
        current = current.next
        
        # If the popped node has a next node, push it to the heap.
        if node.next:
            heapq.heappush(min_heap, (node.next.val, node_idx_counter, node.next))
            node_idx_counter += 1
            
    return dummy.next

if __name__ == "__main__":
    # Helper function to convert a Python list to a linked list.
    def list_to_linkedlist(arr):
        if not arr:
            return None
        head = ListNode(arr[0])
        current = head
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        return head

    # Helper function to convert a linked list to a Python list.
    def linkedlist_to_list(head):
        arr = []
        current = head
        while current:
            arr.append(current.val)
            current = current.next
        return arr

    # Test cases
    
    # Test 1: Basic case with multiple non-empty lists
    lists1 = [list_to_linkedlist([1,4,5]), list_to_linkedlist([1,3,4]), list_to_linkedlist([2,6])]
    expected1 = [1,1,2,3,4,4,5,6]
    result1 = linkedlist_to_list(solution(lists1))
    assert result1 == expected1, f"Test Case 1 Failed: Expected {expected1}, Got {result1}"

    # Test 2: Empty input list (no lists to merge)
    lists2 = []
    expected2 = []
    result2 = linkedlist_to_list(solution(lists2))
    assert result2 == expected2, f"Test Case 2 Failed: Expected {expected2}, Got {result2}"

    # Test 3: List containing only empty linked lists
    lists3 = [list_to_linkedlist([]), list_to_linkedlist([]), list_to_linkedlist([])]
    expected3 = []
    result3 = linkedlist_to_list(solution(lists3))
    assert result3 == expected3, f"Test Case 3 Failed: Expected {expected3}, Got {result3}"

    # Test 4: List with some empty linked lists and some non-empty
    lists4 = [list_to_linkedlist([1,5]), list_to_linkedlist([]), list_to_linkedlist([2,4])]
    expected4 = [1,2,4,5]
    result4 = linkedlist_to_list(solution(lists4))
    assert result4 == expected4, f"Test Case 4 Failed: Expected {expected4}, Got {result4}"

    # Test 5: Single non-empty list in the input
    lists5 = [list_to_linkedlist([1,2,3])]
    expected5 = [1,2,3]
    result5 = linkedlist_to_list(solution(lists5))
    assert result5 == expected5, f"Test Case 5 Failed: Expected {expected5}, Got {result5}"

    # Test 6: All lists are None (explicitly None, not empty list)
    lists6 = [None, None, None]
    expected6 = []
    result6 = linkedlist_to_list(solution(lists6))
    assert result6 == expected6, f"Test Case 6 Failed: Expected {expected6}, Got {result6}"

    # Test 7: Multiple single-node lists
    lists7 = [list_to_linkedlist([1]), list_to_linkedlist([0]), list_to_linkedlist([2])]
    expected7 = [0,1,2]
    result7 = linkedlist_to_list(solution(lists7))
    assert result7 == expected7, f"Test Case 7 Failed: Expected {expected7}, Got {result7}"

    # Test 8: Longer lists with duplicates and negative numbers
    lists8 = [list_to_linkedlist([-10,-9,-9,-7,-5,-4,-2]), list_to_linkedlist([0]), list_to_linkedlist([1,1,3,4,5,6])]
    expected8 = [-10,-9,-9,-7,-5,-4,-2,0,1,1,3,4,5,6]
    result8 = linkedlist_to_list(solution(lists8))
    assert result8 == expected8, f"Test Case 8 Failed: Expected {expected8}, Got {result8}"

    # Test 9: Two lists, one much longer than the other
    lists9 = [list_to_linkedlist([1,2,3,4,5,6,7,8,9,10]), list_to_linkedlist([0])]
    expected9 = [0,1,2,3,4,5,6,7,8,9,10]
    result9 = linkedlist_to_list(solution(lists9))
    assert result9 == expected9, f"Test Case 9 Failed: Expected {expected9}, Got {result9}"

    # Test 10: All lists have one element
    lists10 = [list_to_linkedlist([5]), list_to_linkedlist([1]), list_to_linkedlist([8]), list_to_linkedlist([2])]
    expected10 = [1,2,5,8]
    result10 = linkedlist_to_list(solution(lists10))
    assert result10 == expected10, f"Test Case 10 Failed: Expected {expected10}, Got {result10}"

    print("All test cases passed!")