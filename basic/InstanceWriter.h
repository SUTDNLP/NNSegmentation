#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;

class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
	  if (!m_outf.is_open()) return -1;

	  const vector<string> &labels = pInstance->labels;

	  for (int i = 0; i < labels.size(); ++i) {
	    m_outf << pInstance->words[i] << " ";
	    m_outf << labels[i] << endl;
	  }
	  m_outf << endl;
	  return 0;
	}
};

#endif

