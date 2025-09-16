



pub trait IntoTupleOfVec{
    type    TypleOfVectors;
    fn into_tuple_of_vectors( self ) -> Self::TypleOfVectors;
}

impl <T,U,V> 
    IntoTupleOfVec for

    Vec<((T,U),V)>
{
    type TypleOfVectors     =   (Vec<T>,Vec<U>,Vec<V>);
    
    fn into_tuple_of_vectors( self ) -> Self::TypleOfVectors {
        let (vec_tu, vec_v): (Vec<_>, Vec<_>) = self.into_iter().unzip();
        let (vec_t, vec_u): (Vec<_>, Vec<_>) = vec_tu.into_iter().unzip();
        return (vec_t, vec_u, vec_v)
    }
    
}