#ifndef WEBNN_NATIVE_NAMED_RECORDS_H_
#define WEBNN_NATIVE_NAMED_RECORDS_H_

#include <map>
#include <string>

#include "common/RefCounted.h"

namespace dawn_native {

template <typename T>
class NamedRecords : public RefCounted {
public:
  NamedRecords() = default;
  virtual ~NamedRecords() = default;

  // DAWN API
  void Set(char const * name, const T * record) {
    records_[std::string(name)] = record;
  }

  T* Get(char const * name) const {
    if (records_.find(std::string(name)) == records_.end()) {
      return nullptr;
    }
    return const_cast<T*>(records_.at(std::string(name)));
  }

  // Other methods
  const std::map<std::string, const T*>& GetRecords() const {
    return records_;
  }

private:
  std::map<std::string, const T*> records_;
};
}

#endif  // WEBNN_NATIVE_RECORD_H_