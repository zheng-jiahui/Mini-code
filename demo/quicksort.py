"""
快速排序实现 - 原地分区递归版本
"""


def partition(arr, low, high):
    """
    原地分区函数
    选择最后一个元素作为基准，将小于基准的放左边，大于等于的放右边
    返回基准元素的最终位置
    """
    pivot = arr[high]  # 选择最后一个元素作为基准
    i = low - 1  # i 指向小于 pivot 区域的最后一个元素
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 交换
    
    # 将基准放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort_recursive(arr, low, high):
    """
    递归快速排序 - 原地分区
    """
    if low < high:
        # 分区并获取基准位置
        pi = partition(arr, low, high)
        
        # 递归排序基准左右两部分
        quicksort_recursive(arr, low, pi - 1)
        quicksort_recursive(arr, pi + 1, high)


def quicksort(arr):
    """
    快速排序入口函数
    对列表进行原地排序
    """
    if len(arr) <= 1:
        return arr
    quicksort_recursive(arr, 0, len(arr) - 1)
    return arr


def test_quicksort():
    """自测若干组用例"""
    print("=" * 60)
    print("快速排序自测用例")
    print("=" * 60)
    
    # 用例 1: 空数组
    print("\n[用例 1] 空数组")
    arr1 = []
    print(f"  排序前: {arr1}")
    result1 = quicksort(arr1.copy())
    print(f"  排序后: {result1}")
    assert result1 == [], "空数组测试失败"
    print("  ✓ 通过")
    
    # 用例 2: 单元素
    print("\n[用例 2] 单元素数组")
    arr2 = [42]
    print(f"  排序前: {arr2}")
    result2 = quicksort(arr2.copy())
    print(f"  排序后: {result2}")
    assert result2 == [42], "单元素测试失败"
    print("  ✓ 通过")
    
    # 用例 3: 已排序数组
    print("\n[用例 3] 已排序数组")
    arr3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"  排序前: {arr3}")
    result3 = quicksort(arr3.copy())
    print(f"  排序后: {result3}")
    assert result3 == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "已排序测试失败"
    print("  ✓ 通过")
    
    # 用例 4: 逆序数组
    print("\n[用例 4] 逆序数组")
    arr4 = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    print(f"  排序前: {arr4}")
    result4 = quicksort(arr4.copy())
    print(f"  排序后: {result4}")
    assert result4 == [1, 2, 3, 4, 5, 6, 7, 8, 9], "逆序测试失败"
    print("  ✓ 通过")
    
    # 用例 5: 含重复元素
    print("\n[用例 5] 含重复元素数组")
    arr5 = [5, 2, 8, 2, 9, 1, 5, 5, 3, 8, 2]
    print(f"  排序前: {arr5}")
    result5 = quicksort(arr5.copy())
    print(f"  排序后: {result5}")
    expected5 = [1, 2, 2, 2, 3, 5, 5, 5, 8, 8, 9]
    assert result5 == expected5, "重复元素测试失败"
    print("  ✓ 通过")
    
    # 用例 6: 随机数组
    print("\n[用例 6] 随机数组")
    arr6 = [34, 7, 23, 32, 5, 62, 78, 1, 90, 45]
    print(f"  排序前: {arr6}")
    result6 = quicksort(arr6.copy())
    print(f"  排序后: {result6}")
    expected6 = [1, 5, 7, 23, 32, 34, 45, 62, 78, 90]
    assert result6 == expected6, "随机数组测试失败"
    print("  ✓ 通过")
    
    # 用例 7: 含负数
    print("\n[用例 7] 含负数数组")
    arr7 = [-5, 3, -1, 0, 7, -10, 2, -3]
    print(f"  排序前: {arr7}")
    result7 = quicksort(arr7.copy())
    print(f"  排序后: {result7}")
    expected7 = [-10, -5, -3, -1, 0, 2, 3, 7]
    assert result7 == expected7, "含负数测试失败"
    print("  ✓ 通过")
    
    print("\n" + "=" * 60)
    print("所有测试用例通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_quicksort()
