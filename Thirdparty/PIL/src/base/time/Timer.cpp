
#include <time.h>
#include <sys/time.h>
#include <sstream>
#include <iomanip>
#include <unistd.h>



#include "../Svar/Svar_Inc.h"
#include "Time.h"
#include "Timer.h"


namespace pi {

// Macros for easy access to memory with the correct types:
#ifdef ZY_OS_WINDOWS
#	define	LARGE_INTEGER_NUMS	reinterpret_cast<LARGE_INTEGER*>(largeInts)
#else
#	define	TIMEVAL_NUMS			reinterpret_cast<struct timeval*>(largeInts)
#endif

Timer timer;



void TicTac::Tic()
{
#ifdef ZY_OS_WINDOWS
    LARGE_INTEGER *l= LARGE_INTEGER_NUMS;
    QueryPerformanceCounter(&l[1]);
#else
    struct timeval* ts = TIMEVAL_NUMS;
    gettimeofday( &ts[0], NULL);
#endif
}

double TicTac::Tac()
{
#ifdef ZY_OS_WINDOWS
    LARGE_INTEGER *l= LARGE_INTEGER_NUMS;
    QueryPerformanceCounter( &l[2] );
    return (l[2].QuadPart-l[1].QuadPart)/static_cast<double>(l[0].QuadPart);
#else
    struct timeval* ts = TIMEVAL_NUMS;
    gettimeofday( &ts[1], NULL);

    return ( ts[1].tv_sec - ts[0].tv_sec) +
           1e-6*(  ts[1].tv_usec - ts[0].tv_usec );
#endif
}

Rate::Rate(double frequency)
{
#ifdef ZY_OS_WINDOWS
#else
    cycle=1.0/frequency;
    struct timeval now;
    gettimeofday(&now,NULL);
    last_start=now.tv_sec+now.tv_usec/1000000.0;
#endif
}

bool Rate::sleep()
{
#ifdef ZY_OS_WINDOWS
#else
    struct timeval now;
    gettimeofday(&now, NULL);
    double slp=cycle-(now.tv_sec+now.tv_usec/1000000.0-last_start);
//    struct timespec ts = { (int)slp,((int)(slp*1000000))%1000000000 };
//    nanosleep(&ts, 0);
    if(slp>0)
        usleep(slp*1000000);
    gettimeofday(&now,NULL);
    last_start=now.tv_sec+now.tv_usec/1000000.0;
#endif
    return 0;
}


Timer::TCallData::TCallData() :
    n_calls	(0),
    min_t	(0),
    max_t	(0),
    mean_t	(0),
    has_time_units(true)
{

}

Timer::~Timer()
{
    if(SvarWithType<int>::instance().get_var("Timer.DumpAllStats",1))
    {
        dumpAllStats();
        SvarWithType<int>::instance()["Timer.DumpAllStats"]=0;
    }
}


void Timer::do_enter(const char *func_name)
{
    const string  s = func_name;
    TCallData &d = m_data[s];

    d.n_calls++;
    d.open_calls.push(0);  // Dummy value, it'll be written below
    d.open_calls.top() =Tac(); // to avoid possible delays.
}

double Timer::do_leave(const char *func_name)
{
    const double tim = Tac();

    const string  s = func_name;
    TCallData &d = m_data[s];

    if (!d.open_calls.empty())
    {
        const double At = tim - d.open_calls.top();
        d.open_calls.pop();

        d.mean_t+=At;
        if (d.n_calls==1)
        {
            d.min_t= At;
            d.max_t= At;
        }
        else
        {
            if (d.min_t>At) d.min_t = At;
            if (d.max_t<At) d.max_t = At;
        }
        return At;
    }
    else return 0; // This shouldn't happen!
}

double Timer::getMeanTime(const std::string &name)  const
{
    map<string,TCallData>::const_iterator it = m_data.find(name);
    if (it==m_data.end())
         return 0;
    else return it->second.n_calls ? it->second.mean_t/it->second.n_calls : 0;
}

std::string unitsFormat(const double val,int nDecimalDigits, bool middle_space)
{
    char	prefix;
    double	mult;

    if (val>=1e12)
        {mult=1e-12; prefix='T';}
    else if (val>=1e9)
        {mult=1e-9; prefix='G';}
    else if (val>=1e6)
        {mult=1e-6; prefix='M';}
    else if (val>=1e3)
        {mult=1e-3; prefix='K';}
    else if (val>=1)
        {mult=1; prefix=' ';}
    else if (val>=1e-3)
        {mult=1e+3; prefix='m';}
    else if (val>=1e-6)
        {mult=1e+6; prefix='u';}
    else if (val>=1e-9)
        {mult=1e+9; prefix='n';}
    else if (val>=1e-12)
        {mult=1e+12; prefix='p';}
    else
        {mult=0; prefix='p';}
    ostringstream ost;
    ost<<setw(5) <<setiosflags(ios::fixed) <<setiosflags(ios::right)
      << setprecision(1)<<(val*mult);
    return ost.str()+char(prefix);
}

std::string rightPad(const std::string &str, const size_t total_len, bool truncate_if_larger)
{
    std::string r = str;
    if (r.size()<total_len || truncate_if_larger)
        r.resize(total_len,' ');
    return r;
}

std::string  aux_format_string_multilines(const std::string &s, const size_t len)
{
    std::string ret;

    for (size_t p=0;p<s.size();p+=len)
    {
        ret+=rightPad(s.c_str()+p,len,true);
        if (p+len<s.size())
            ret+="\n";
    }
    return ret;
}

std::string Timer::getStatsAsText(const size_t column_width)  const
{
    ostringstream ost;
    ost<<"---------------------------- ZhaoYong::Timer report --------------------------\n";
    ost<<"           FUNCTION                       #CALLS  MIN.T  MEAN.T  MAX.T  TOTAL \n";
    ost<<"------------------------------------------------------------------------------\n";
    for (map<string,TCallData>::const_iterator i=m_data.begin();i!=m_data.end();++i)
    {
        const string sMinT   = unitsFormat(i->second.min_t,1,false);
        const string sMaxT   = unitsFormat(i->second.max_t,1,false);
        const string sTotalT = unitsFormat(i->second.mean_t,1,false);
        const string sMeanT  = unitsFormat(i->second.n_calls ? i->second.mean_t/i->second.n_calls : 0,1,false);

        ost<<aux_format_string_multilines(i->first,39)
          <<" "<<setw(6)<<setiosflags(ios::right)<<i->second.n_calls<<"  "
         <<sMinT<<"s "<<sMeanT<<"s "<<sMaxT<<"s "
        <<sTotalT<<"s\n";
    }

    ost<<"----------------------- End of ZhaoYong::Timer report ------------------------\n";

    return ost.str();
}

void Timer::dumpAllStats(const size_t  column_width) const
{
    if(!m_data.size()) return;
    string s = getStatsAsText(column_width);
    cout<<endl<<s<<endl;
}

} // end of namespace pi
