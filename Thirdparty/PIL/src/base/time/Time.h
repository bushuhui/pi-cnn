#ifndef __TIME_H__
#define __TIME_H__

#include "base/types/types.h"

namespace pi {


////////////////////////////////////////////////////////////////////////////////
/// time functions
////////////////////////////////////////////////////////////////////////////////

///
/// \brief get mil-second
/// \return mil-second (unsigned 64-bit interger)
///
ru64 tm_get_millis(void);

///
/// \brief get mil-second
/// \return mil-second (unsigned 64-bit interger)
///
ru64 tm_get_ms(void);

///
/// \brief get micro second
/// \return micro-second (unsigned 64-bit interger)
///
ru64 tm_get_us(void);

///
/// \brief get time stamp
///
/// \return second since 1970-1-1
///
double tm_getTimeStamp(void);

uint32_t tm_getTimeStampUnix(void);


///
/// \brief sleep a mil-second
/// \param t - mil-second (unsigned 32-bit interger)
///
void tm_sleep(ru32 t);

///
/// \brief sleep a micro-second
/// \param t - micro-second (unsigned 32-bit interger)
///
void tm_sleep_us(ri64 t);


} // end of namespace pi


#endif // end of __TIME_H__
