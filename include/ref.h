
#include <type_traits>

#ifndef REF_REF_H
#define REF_REF_H

template<typename T>
std::enable_if_t<std::is_pointer<T>::value, std::remove_pointer_t<T>&> deref(T& t) {
    return *t;
}

template<typename T>
std::enable_if_t<!std::is_pointer<T>::value, T&> deref(T& t) {
    return t;
}

#endif //REF_REF_H
