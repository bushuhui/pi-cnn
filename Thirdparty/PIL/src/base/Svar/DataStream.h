/******************************************************************************

  Robot Toolkit ++ (RTK++)

  Copyright (c) 2007-2013 Shuhui Bu <bushuhui@nwpu.edu.cn>
  http://www.adv-ci.com

  ----------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/

#ifndef __RTK_DATASTREAM_H__
#define __RTK_DATASTREAM_H__

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>
#include <ostream>

#include "base/types/types.h"
#include "base/debug/debug_config.h"

namespace pi {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline void RDataStream_memcpy(void *dst, const void *src, int len)
{
    ru8     *pd = (ru8*) dst, 
            *ps = (ru8*) src;
    
    for(int i=0; i<len; i++) pd[i] = ps[i];
}
    


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

///
/// \brief The DataStream class
///
class RDataStream
{
public:
    ///
    /// \brief RDataStream - default initialize
    ///
    RDataStream() {
        m_fromRawData = 0;
        init();
    }

    ///
    /// \brief RDataStream - set to given length
    /// \param length - stream length
    ///
    RDataStream(ru32 length) {
        m_fromRawData = 0;
        init();
        resize(length);
    }

    ///
    /// \brief RDataStream - from raw data (read-only)
    /// \param d    - byte array
    /// \param l    - array size
    ///
    RDataStream(ru8 *d, ru32 l) {
        m_fromRawData = 1;
        m_arrData = NULL;
        fromRawData_noCopy(d, l);
    }

    ///
    /// \brief RDataStream - from string (read-only)
    /// \param dat  - raw data encapsuled to string
    ///
    RDataStream(std::string &dat) {
        m_fromRawData = 1;
        m_arrData = NULL;
        fromRawData_noCopy((ru8*) dat.c_str(), dat.size());
    }

    ///
    /// \brief ~RDataStream
    ///
    virtual ~RDataStream() {
        if( !m_fromRawData )
            release();
    }

    ///
    /// \brief clear all data
    ///
    void clear(void) {
        clear_noDel();
    }

	///
	/// \brief clear without delete buffer
	///
    void clear_noDel(void) {
        if( m_fromRawData ) {
            dbg_pe("Datastream from raw data! Please do not modify it\n");
            exit(1);
        }

        m_size = 2*sizeof(ru32);
        m_idx  = 2*sizeof(ru32);
    }


    ///
    /// \brief resize stream size
    /// \param n    - new size
    /// \param c    - default byte value
    /// \return
    ///
    int resize(int n, ri32 c=-1) {
        if( m_fromRawData ) {
            dbg_pe("Datastream from raw data! Please do not modify it\n");
            exit(1);
        }

        if( n == m_size ) {
            return 0;
        } else if ( n < m_size ) {
            if( n < 2*sizeof(ru32) ) n = 2*sizeof(ru32);
            if( m_idx > n ) m_idx = n;
        } else {
            if( n > m_sizeReserved ) {
                int nSizeRev = n*2;
                reserve(nSizeRev, c);
            }
        }
        
        m_size = n;
        updateHeader();

        return 0;
    }

    ///
    /// \brief reserve given length's buffer
    /// \param n    - buffer size
    /// \param c    - default byte value
    /// \return
    ///
    int reserve(int n, ri32 c=-1) {
        if( m_fromRawData ) {
            dbg_pe("Datastream from raw data! Please do not modify it\n");
            exit(1);
        }

        if( n <= m_sizeReserved ) {
            return -1;
        }

        // alloc new buffer
        ru8* arrN, cb;
        arrN = new ru8[n];

        if( m_idx > 0 ) {
            RDataStream_memcpy(arrN, m_arrData, m_idx);

            // set new alloc data to given c
            if( c >= 0 ) {
                cb = c;
                for(int i=m_idx; i<n; i++) arrN[i] = cb;
            }

            // release old data
            delete [] m_arrData;
        }

        // set data pointers
        m_arrData = arrN;
        m_sizeReserved = n;

        return 0;
    }

    ///
    /// \brief set RDataStream to given raw data
    /// \param d    - raw data byte array
    /// \param l    - array size
    /// \return     - this object
    ///
    RDataStream& fromRawData(ru8 *d, ru32 l) {
        if( l > m_sizeReserved ) reserve(l);

        RDataStream_memcpy(m_arrData, d, l);
        m_size = l;
        m_idx = 2*sizeof(ru32);

        return *this;
    }

    ///
    /// \brief set RDataStream to given raw data (without copy)
    /// \param d    - raw data byte array
    /// \param l    - array size
    /// \return     - this object
    ///
    RDataStream& fromRawData_noCopy(ru8 *d, ru32 l) {
        if( !m_fromRawData ) {
            if( m_arrData != NULL ) delete [] m_arrData;
        }

        m_fromRawData = 1;
        m_arrData = d;
        m_size = l;
        m_sizeReserved = l;

        m_idx = 2*sizeof(ru32);

        return *this;
    }


    ///
    /// \brief rewind to beginning position
    /// \return
    ///
    int rewind(void) {
        m_idx = 2*sizeof(ru32);
        return 0;
    }

    ///
    /// \brief seek pointer position
    /// \param offset       - offset
    /// \param whence       - baseline position
    ///                         SEEK_SET - beginning position
    ///                         SEEK_CUR - current position
    ///                         SEEK_END - end of stream
    /// \return
    ///     0   - success
    ///     -1  - given position outside the stream
    ///
    int seek(ri32 offset, int whence=SEEK_SET) {
        ri32 np, bp;

        if( whence == SEEK_SET ) bp = 0;
        if( whence == SEEK_CUR ) bp = m_idx;
        if( whence == SEEK_END ) bp = m_size;

        np = bp + offset;
        if( np > m_size ) return -1;
        if( np < 2*sizeof(ru32) ) np = 2*sizeof(ru32);

        m_idx = np;
        return 0;
    }

    ///
    /// \brief get the size of stream
    /// \return stream length
    ///
    /// \see length()
    ///
    ru32 size(void) {
        return m_size;
    }

    ///
    /// \brief get the length of stream
    /// \return stream length
    ///
    /// \see size()
    ///
    ru32 length(void) {
        return m_size;
    }

    ///
    /// \brief return stream raw data
    /// \return - raw data pointer
    ///
    ru8* data(void) {
        return m_arrData;
    }

    ///
    /// \brief set RDataStream header
    ///
    /// \param magic    - magic number
    /// \param ver      - version
    ///
    /// \return
    ///     0           - success
    ///
    int setHeader(ru32 magic, ru32 ver) {
        ru32    mv;

        mv = (ver << 16) | (magic & 0x0000FFFF);
        RDataStream_memcpy(m_arrData, &mv, sizeof(ru32));

        return 0;
    }

    ///
    /// \brief get RDataStream header
    ///
    /// \param magic    - magic number
    /// \param ver      - version
    ///
    /// \return
    ///     0           - success
    ///
    int getHeader(ru32& magic, ru32& ver) {
        ru32    mv;

        RDataStream_memcpy(&mv, m_arrData, sizeof(ru32));

        magic   = mv & 0x0000FFFF;
        ver     = mv >> 16;

        return 0;
    }

    ///
    /// \brief setVerNum
    /// \param ver - version number
    /// \return
    ///
    int setVerNum(ru32 magic) {
        ru32 mv;
        ru8* p;

        p = m_arrData;
        RDataStream_memcpy(&mv, p, sizeof(ru32));

        mv = mv & 0x0000FFFF;
        mv = mv | (magic << 16);
        RDataStream_memcpy(p, &mv, sizeof(ru32));

        return 0;
    }

    ///
    /// \brief getVerNum
    /// \param ver - version number
    /// \return
    ///
    int getVerNum(ru32& magic) {
        ru32 mv;
        ru8* p;

        p = m_arrData;
        RDataStream_memcpy(&mv, p, sizeof(ru32));

        magic = mv >> 16;

        return 0;
    }

    ///
    /// \brief setMagicNum
    /// \param magic - magic number
    /// \return
    ///
    int setMagicNum(ru32 ver) {
        ru32 mv;
        ru8* p;

        p = m_arrData;
        RDataStream_memcpy(&mv, p, sizeof(ru32));

        mv = mv & 0xFFFF0000;
        mv = mv | (ver & 0x0000FFFF);
        RDataStream_memcpy(p, &mv, sizeof(ru32));

        return 0;
    }

    ///
    /// \brief getMagicNum
    /// \param magic - magic number
    /// \return
    ///
    int getMagicNum(ru32& ver) {
        ru32 mv;
        ru8* p;

        p = m_arrData;
        RDataStream_memcpy(&mv, p, sizeof(ru32));

        ver = mv & 0x0000FFFF;

        return 0;
    }


    ///
    /// \brief update stream header
    /// \return
    ///
    int updateHeader(void) {
        ru8* p;

        p = m_arrData + sizeof(ru32);
        RDataStream_memcpy(p, &m_size, sizeof(ru32));

        return 0;
    }



public:

    #define write_(d) \
        ru32 sn, dl; \
        dl = sizeof(d); \
        sn = m_idx + dl; \
        if( sn > m_sizeReserved ) reserve(sn*2); \
        RDataStream_memcpy((m_arrData+m_idx), &d, dl); \
        m_idx += dl; \
        if( m_idx > m_size ) m_size = m_idx; \
        updateHeader(); \
        return 0;

    ///
    /// \brief write ri8 data
    /// \param d - ri8 data
    /// \return
    ///
    int write(ri8 &d) {
        write_(d);
    }

    ///
    /// \brief write ru8 data
    /// \param d - ru8 data
    /// \return
    ///
    int write(ru8 &d) {
        write_(d);
    }

    ///
    /// \brief write ri16 data
    /// \param d - ri16 data
    /// \return
    ///
    int write(ri16 &d) {
        write_(d);
    }


    ///
    /// \brief write ru16 data
    /// \param d - ru16 data
    /// \return
    ///
    int write(ru16 &d) {
        write_(d);
    }

    ///
    /// \brief write ri32 data
    /// \param d
    /// \return
    ///
    int write(ri32 &d) {
        write_(d);
    }


    ///
    /// \brief write ru32 data
    /// \param d
    /// \return
    ///
    int write(ru32 &d) {
        write_(d);
    }

    ///
    /// \brief write ri64 data
    /// \param d
    /// \return
    ///
    int write(ri64 &d) {
        write_(d);
    }

    ///
    /// \brief write ru64 data
    /// \param d
    /// \return
    ///
    int write(ru64 &d) {
        write_(d);
    }

    ///
    /// \brief write rf32 (float) data
    /// \param d
    /// \return
    ///
    int write(rf32 &d) {
        write_(d);
    }

    ///
    /// \brief write rf64 (double) data
    /// \param d
    /// \return
    ///
    int write(rf64 &d) {
        write_(d);
    }

    ///
    /// \brief write binary data
    /// \param d   - binary data array
    /// \param len - length in byte
    /// \return
    ///
    int write(ru8 *d, ru32 len) {
        ru32 sn, dl;

        dl = len;
        sn = m_idx + dl;
        if( sn > m_sizeReserved ) reserve(sn*2);

        RDataStream_memcpy((m_arrData+m_idx), d, dl);

        m_idx += dl;
        if( m_idx > m_size ) m_size = m_idx;
        updateHeader();

        return 0;
    }

    ///
    /// \brief write a std::string obj
    /// \param s - std::string obj
    ///
    /// \return
    ///     0           - success
    ///
    int write(std::string &s) {
        ru32    sl;
        ru32    sn, dl;

        // determine length
        sl = s.size();
        dl = sizeof(ru32) + sl + 1;
        sn = m_idx + dl;
        if( sn > m_sizeReserved ) reserve(sn*2);

        // copy data
        RDataStream_memcpy((m_arrData+m_idx), &sl, sizeof(ru32));
        RDataStream_memcpy((m_arrData+m_idx+sizeof(ru32)), (void*) s.c_str(), sl);
        m_arrData[m_idx + sizeof(ru32) + sl] = 0;

        // update index
        m_idx += dl;
        if( m_idx > m_size ) m_size = m_idx;
        updateHeader();

        return 0;
    }

    ///
    /// \brief write RDataStream data
    /// \param d - RDataStream data
    /// \return
    ///
    int write(RDataStream &d) {
        ru32 sn, dl;

        dl = d.size();
        sn = m_idx + dl;
        if( sn > m_sizeReserved ) reserve(sn*2);

        RDataStream_memcpy((m_arrData+m_idx), d.data(), dl);

        m_idx += dl;
        if( m_idx > m_size ) m_size = m_idx;
        updateHeader();

        return 0;
    }



    #define read_(d) \
        ru32 sn, dl; \
        dl = sizeof(d); \
        sn = m_idx + dl; \
        if( sn > m_size ) return -1; \
        RDataStream_memcpy(&d, (m_arrData+m_idx), dl); \
        m_idx += dl; \
        return 0; \


    ///
    /// \brief read ri8 data
    /// \param d
    /// \return
    ///
    int read(ri8 &d) {
        read_(d);
    }

    ///
    /// \brief read ru8 data
    /// \param d
    /// \return
    ///
    int read(ru8 &d) {
        read_(d);
    }

    ///
    /// \brief read ri16 data
    /// \param d
    /// \return
    ///
    int read(ri16 &d) {
        read_(d);
    }

    ///
    /// \brief read ru16 data
    /// \param d
    /// \return
    ///
    int read(ru16 &d) {
        read_(d);
    }

    ///
    /// \brief read ri32 data
    /// \param d
    /// \return
    ///
    int read(ri32 &d) {
        read_(d);
    }

    ///
    /// \brief read ru32 data
    /// \param d
    /// \return
    ///
    int read(ru32 &d) {
        read_(d);
    }

    ///
    /// \brief read ri64 data
    /// \param d
    /// \return
    ///
    int read(ri64 &d) {
        read_(d);
    }

    ///
    /// \brief read ru64 data
    /// \param d
    /// \return
    ///
    int read(ru64 &d) {
        read_(d);
    }

    ///
    /// \brief read rf32 (float) data
    /// \param d
    /// \return
    ///
    int read(rf32 &d) {
        read_(d);
    }

    ///
    /// \brief read rf64 (double) data
    /// \param d
    /// \return
    ///
    int read(rf64 &d) {
        read_(d);
    }

    ///
    /// \brief read binary data
    /// \param d   - binar data array
    /// \param len - length in byte
    /// \return
    ///
    int read(ru8 *d, int len) {
        ru32 sn, dl;

        dl = len;
        sn = m_idx + dl;
        if( sn > m_size ) return -1;

        RDataStream_memcpy(d, (m_arrData+m_idx), dl);

        m_idx += dl;

        return 0;
    }

    ///
    /// \brief read a std::string obj
    /// \param s - std::string obj
    ///
    /// \return
    ///     0       - success
    ///     -1      - out of index
    ///
    int read(std::string &s) {
        ru32    sl;
        ru32    sn, dl;
        char    *buf;

        // read string length
        dl = sizeof(ru32);
        sn = m_idx + dl;
        if( sn > m_size ) return -1;
        RDataStream_memcpy(&sl, (m_arrData+m_idx), dl);

        // determine total length
        sn = m_idx + sizeof(ru32) + sl;
        if( sn > m_size ) return -1;

        // read string
        buf = (char*) (m_arrData+m_idx+sizeof(ru32));
        s = buf;

        // update index
        m_idx += sizeof(ru32) + sl + 1;

        return 0;
    }

    ///
    /// \brief read RDataStream data
    /// \param d - RDataStream data
    /// \return
    ///
    int read(RDataStream &d) {
        ru32 hl, sn, dl;

        // read header information
        hl = 2*sizeof(ru32);
        if( m_idx + hl > m_size ) return -1;
        RDataStream_memcpy(&dl, (m_arrData+m_idx+sizeof(ru32)), sizeof(ru32));

        // check length
        sn = m_idx + dl;
        if( sn > m_size ) return -1;

        // read data
        d.fromRawData(m_arrData+m_idx, dl);
        m_idx += dl;

        return 0;
    }

    ///
    /// \brief read RDataStream data (fast without memcpy)
    /// \param d - RDataStream data
    /// \return
    ///
    int readFast(RDataStream &d) {
        ru32 hl, sn, dl;

        // read header information
        hl = 2*sizeof(ru32);
        if( m_idx + hl > m_size ) return -1;
        RDataStream_memcpy(&dl, (m_arrData+m_idx+sizeof(ru32)), sizeof(ru32));

        // check length
        sn = m_idx + dl;
        if( sn > m_size ) return -1;

        // read data
        d.fromRawData_noCopy(m_arrData+m_idx, dl);
        m_idx += dl;

        return 0;
    }


    RDataStream& operator << (ri8 &d) {
        write(d);
        return *this;
    }


    RDataStream& operator << (ru8 &d) {
        write(d);
        return *this;
    }

    RDataStream& operator << (ri16 &d) {
        write(d);
        return *this;
    }


    RDataStream& operator << (ru16 &d) {
        write(d);
        return *this;
    }

    RDataStream& operator << (ri32 &d) {
        write(d);
        return *this;
    }


    RDataStream& operator << (ru32 &d) {
        write(d);
        return *this;
    }

    RDataStream& operator << (ri64 &d) {
        write(d);
        return *this;
    }


    RDataStream& operator << (ru64 &d) {
        write(d);
        return *this;
    }

    RDataStream& operator << (rf32 &d) {
        write(d);
        return *this;
    }


    RDataStream& operator << (rf64 &d) {
        write(d);
        return *this;
    }

    RDataStream& operator << (std::string &s) {
        write(s);
        return *this;
    }

    RDataStream& operator << (RDataStream &d) {
        write(d);
        return *this;
    }


    RDataStream& operator >> (ri8 &d) {
        read(d);
        return *this;
    }


    RDataStream& operator >> (ru8 &d) {
        read(d);
        return *this;
    }

    RDataStream& operator >> (ri16 &d) {
        read(d);
        return *this;
    }


    RDataStream& operator >> (ru16 &d) {
        read(d);
        return *this;
    }

    RDataStream& operator >> (ri32 &d) {
        read(d);
        return *this;
    }


    RDataStream& operator >> (ru32 &d) {
        read(d);
        return *this;
    }

    RDataStream& operator >> (ri64 &d) {
        read(d);
        return *this;
    }


    RDataStream& operator >> (ru64 &d) {
        read(d);
        return *this;
    }

    RDataStream& operator >> (rf32 &d) {
        read(d);
        return *this;
    }


    RDataStream& operator >> (rf64 &d) {
        read(d);
        return *this;
    }

    RDataStream& operator >> (std::string &s) {
        read(s);
        return *this;
    }

    RDataStream& operator >> (RDataStream &d) {
        read(d);
        return *this;
    }


    void print(void) {
        ru32    magic, ver;

        getHeader(magic, ver);
        printf("DataStream: magic = 0x%04X, ver = %d, size = %d\n content = ", magic, ver, m_size);
        for(int i=2*sizeof(ru32); i<m_size; i++) {
            printf("0x%02X ", m_arrData[i]);
        }
        printf("\n");
    }

protected:

    ///
    /// \brief init RDataStream obj
    ///
    void init(void) {
        if( m_fromRawData ) {
            dbg_pe("Datastream from raw data! Please do not modify it\n");
            exit(1);
        }

        m_fromRawData = 0;
        m_sizeReserved = 2048;

        m_arrData = new ru8[m_sizeReserved];
        for(int i=0; i<m_sizeReserved; i++) m_arrData[i] = 0;

        m_size = 2*sizeof(ru32);
        m_idx  = 2*sizeof(ru32);
    }

    ///
    /// \brief release RDataStream obj
    ///
    void release(void) {
        if( m_fromRawData ) {
            dbg_pe("Datastream from raw data! Please do not modify it\n");
            exit(1);
        }

        if( m_arrData != NULL ) {
            delete [] m_arrData;
            m_arrData = NULL;
        }

        m_idx = 0;
        m_sizeReserved = 0;
        m_size = 0;
    }

protected:
    ru8*                m_arrData;                  ///< data array
    ru32                m_idx;                      ///< current position
    ru32                m_size;                     ///< current length
    ru32                m_sizeReserved;             ///< stream actual memory size (reserved)

    int                 m_fromRawData;              ///< datastream from raw data
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline int datastream_get_header(ru8 *buf, ru32 &magic, ru32 &ver)
{
    ru32 mv;

    memcpy(&mv, buf, sizeof(ru32));

    magic   = mv & 0x0000FFFF;
    ver     = mv >> 16;

    return 0;
}

inline ru32 datastream_get_length(ru8 *buf)
{
    ru32 len;

    memcpy(&len, buf+4, sizeof(ru32));

    return len;
}

inline int datastream_set_header(ru8 *buf, ru32 magic, ru32 ver, ru32 len)
{
    ru32 mv;

    mv = ver << 16 | (magic & 0x0000FFFF);

    memcpy(buf,   &mv,  sizeof(ru32));
    memcpy(buf+4, &len, sizeof(ru32));

    return 0;
}

} // end of namespace pi

#endif // __RTK_DATASTREAM_H__

