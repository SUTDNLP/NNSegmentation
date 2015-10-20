#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
  InstanceReader() {
  }
  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

    int length = vecLine.size();

    m_instance.allocate(length);

    for (int i = 0; i < length; ++i) {
      vector<string> vecInfo;
      split_bychar(vecLine[i], vecInfo, ' ');
      int veclength = vecInfo.size();
      m_instance.labels[i] = vecInfo[veclength - 1];
      m_instance.words[i] = vecInfo[0];
      for (int j = 1; j < veclength - 1; j++) {
        if (is_startwith(vecInfo[j], "[S]"))
          m_instance.sparsefeatures[i].push_back(vecInfo[j]);
        if (is_startwith(vecInfo[j], "[T"))
          m_instance.tagfeatures[i].push_back(vecInfo[j]);
      }
    }

    return &m_instance;
  }
};

#endif

