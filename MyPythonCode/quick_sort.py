def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr)/2]                 #   pick the middle element
    left = [x for x in arr if x < pivot]    #   create left array with x when it is smaller than pivot
    middle = [x for x in arr if x == pivot] #   these element are already sorted
    right = [x for x in arr if x > pivot]   #   create right array with x when it is greater than pivot

    return quick_sort(left) + middle + quick_sort(right) #recursively call quck sort for left and right , add middle element as it is

print(quick_sort([23, 5, 90, 76, 2, 0]))


