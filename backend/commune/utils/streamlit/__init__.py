import streamlit as st

def describe(module =None, sidebar = True, detail=False, expand=True):
    
    _st = st.sidebar if sidebar else st
    st.sidebar.markdown('# '+str(module))
    fn_list = list(filter(lambda fn: callable(getattr(module,fn)) and '__' not in fn,  dir(module)))
    
    def content_fn(fn_list=fn_list):
        fn_list = _st.multiselect('fns', fn_list)
        for fn_key in fn_list:
            fn = getattr(module,fn_key)
            if callable(fn):
                _st.markdown('#### '+fn_key)
                _st.write(fn)
                _st.write(type(fn))
    if expand:
        with st.sidebar.expander(str(module)):
            content_fn()
    else:
        content_fn()