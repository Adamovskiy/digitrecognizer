package info.adamovskiy.nn.utils;

import java.util.AbstractSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * Fully copied implementation of {@link java.util.HashSet} with one change:
 * there are Integer element indexes contained as backed hash table values.
 * 
 * @param <E>
 *            the type of elements maintained by this set
 */
public class IndexedHashSet<E> extends AbstractSet<E> implements Set<E> {

	private HashMap<E, Integer> map;

	/**
	 * See {@link HashSet#HashSet()} description.
	 */
	public IndexedHashSet() {
		map = new HashMap<>();
	}

	/**
	 * See {@link HashSet#HashSet(java.util.Map)} description.
	 */
	public IndexedHashSet(Collection<? extends E> c) {
		map = new HashMap<>(Math.max((int) (c.size() / .75f) + 1, 16));
		addAll(c);
	}

	/**
	 * See {@link HashSet#HashSet(int, float)} description.
	 */
	public IndexedHashSet(int initialCapacity, float loadFactor) {
		map = new HashMap<>(initialCapacity, loadFactor);
	}

	/**
	 * See {@link HashSet#HashSet(int)} description.
	 */
	public IndexedHashSet(int initialCapacity) {
		map = new HashMap<>(initialCapacity);
	}

	/**
	 * See {@link HashSet#iterator()} description.
	 */
	public Iterator<E> iterator() {
		return map.keySet().iterator();
	}

	/**
	 * See {@link HashSet#size()} description.
	 */
	public int size() {
		return map.size();
	}

	/**
	 * See {@link HashSet#isEmpty()} description.
	 */
	public boolean isEmpty() {
		return map.isEmpty();
	}

	/**
	 * See {@link HashSet#contains(Object)} description.
	 */
	public boolean contains(Object o) {
		return map.containsKey(o);
	}

	/**
	 * See {@link HashSet#add(Object)} description.
	 */
	public boolean add(E e) {
		if (contains(e))
			return false;
		return map.put(e, map.size()) == null;
	}

	/**
	 * See {@link HashSet#remove(Object)} description.
	 */
	public boolean remove(Object o) {
		final Integer removedIndex = map.remove(o);
		final boolean result = removedIndex != null;
		if (result) {
			for (E element : map.keySet()) {
				int currentIndex = map.get(element);
				if (currentIndex > removedIndex)
					map.put(element, currentIndex - 1);
			}
		}
		return result;
	}

	/**
	 * See {@link HashSet#clear()} description.
	 */
	public void clear() {
		map.clear();
	}
	
	/**
	 * Adds e to this set if it is not already in this set (unlike {@link #getIndex(Object)}), returns index of e.
	 * 
	 * @param e element of set
	 * @return index of e
	 */
	public Integer index(E e) {
		if (contains(e))
			return map.get(e);
		int index = map.size();
		map.put(e, index);
		return index;
	}
	
	/**
	 * Gets index of e in this set without adding it (unlike {@link #index(Object)})
	 * 
	 * @param e element of set
	 * @return index of e, if it is in this set, null otherwise
	 */
	public Integer getIndex(E e) {
		return map.get(e);
	}
}
